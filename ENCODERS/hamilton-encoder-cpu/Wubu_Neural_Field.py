# Wubu_Neural_Field.py
# THE GEOMETRIC BRAIN v5.1 [FIXED SCAN]
#
# PAPER REFERENCE: "Neural Field Models: A mathematical overview and unifying framework" (Cook et al., 2022)
# FIX: Corrected nn.scan return signature tuple (carry, output).

import os
import sys
import time
import math
import pickle
import traceback
import multiprocessing as mp
import requests
import numpy as np
import shutil
from functools import partial
from typing import Any, Tuple, List

# --- MEMORY SETTINGS ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
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
# 1. HYPER-PARAMETERS & CONSTANTS
# ==============================================================================
class Config:
    # NEURAL FIELD GEOMETRY
    NUM_NEURONS = 2048      
    LOBE_SEP = 0.6          
    
    # NEURAL FIELD DYNAMICS
    AXONAL_RANGE = 0.4      
    DELAY_VELOCITY = 10.0   
    
    # SEQUENCE LEARNING
    SEQ_LEN = 128           
    D_MODEL = 2048          
    LAYERS = 2              
    
    # OPTIMIZATION
    LR = 2e-4               
    BATCH_SIZE = 1          
    
    # REFLECTION
    USE_SMITH_GATING = True
    GAMMA_PENALTY = 0.5
    
    # I/O
    VOCAB_SIZE = 8192

# ==============================================================================
# 2. THE GEOMETRIC SUBSTRATE: FIBONACCI LATTICE
# ==============================================================================
def generate_fibonacci_brain():
    points = []
    n = Config.NUM_NEURONS
    half = n // 2
    phi = math.pi * (3. - math.sqrt(5.))
    
    for i in range(half):
        y = 1 - (i / float(half - 1)) * 2 
        radius = math.sqrt(1 - y * y)
        theta = phi * i 
        x = math.cos(theta) * radius * 0.8
        z = math.sin(theta) * radius * 0.9
        points.append((x - Config.LOBE_SEP, y, z))

    for i in range(half):
        y = 1 - (i / float(half - 1)) * 2 
        radius = math.sqrt(1 - y * y)
        theta = phi * i 
        x = math.cos(theta) * radius * 0.8
        z = math.sin(theta) * radius * 0.9
        points.append((x + Config.LOBE_SEP, y, z))
        
    points_np = np.array(points, dtype=np.float32)
    dists = np.linalg.norm(points_np[:, None] - points_np[None, :], axis=-1)
    
    # Spatial Kernel (Exponential Decay)
    spatial_kernel = np.exp(-dists / Config.AXONAL_RANGE)
    spatial_kernel = spatial_kernel / (np.sum(spatial_kernel, axis=-1, keepdims=True) + 1e-6)
    
    wires = []
    for i in range(n):
        search_range = 10 
        start = max(0, i - search_range)
        end = min(n, i + search_range)
        local_dists = dists[i, start:end]
        nearest_indices = np.argsort(local_dists)
        for k in range(1, 3): 
            if k < len(nearest_indices):
                target = start + nearest_indices[k]
                if i < target: wires.append((i, target))
            
    return points_np, wires, jnp.array(spatial_kernel)

POINTS_3D, WIRES, SPATIAL_KERNEL_GPU = generate_fibonacci_brain()

# ==============================================================================
# 3. NEURAL FIELD DYNAMICS (FLAX)
# ==============================================================================

def smith_transform(real_h, imag_h):
    r = jax.nn.softplus(real_h) 
    x = imag_h
    denom = (r + 1.0)**2 + x**2
    u = (r**2 + x**2 - 1.0) / denom
    v = (2.0 * x) / denom
    gamma_mag = jnp.sqrt(u**2 + v**2)
    return u, v, gamma_mag

class NeuralFieldLayer(nn.Module):
    """
    Implements the Amari Equation Update Step.
    """
    config: Config
    
    @nn.compact
    def __call__(self, u, _): # _ is the scan input (unused here)
        # u shape: [batch, seq_len, num_neurons]
        
        # 1. Firing Rate f[u]
        firing_rate = nn.sigmoid(u) 
        
        # 2. Spatial Convolution
        synaptic_input = jnp.einsum('bln,nm->blm', firing_rate, SPATIAL_KERNEL_GPU)
        
        # 3. Plasticity
        plasticity = nn.Dense(Config.NUM_NEURONS, use_bias=False, kernel_init=nn.initializers.zeros)
        weighted_input = plasticity(synaptic_input)
        
        # 4. Update
        ff_gate = nn.Dense(Config.D_MODEL)(u)
        ff_gate = nn.gelu(ff_gate)
        
        d_u = -u + weighted_input + ff_gate
        u_next = nn.LayerNorm()(u + d_u)
        
        # RETURN TUPLE: (Carry, Output)
        # Carry = u_next (goes to next layer)
        # Output = None (we don't need intermediate layer outputs)
        return u_next, None 

class WubuNeuralField(nn.Module):
    vocab_size: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Embed(self.vocab_size, Config.D_MODEL)(x)
        
        pos = jnp.arange(0, x.shape[1])[None, :, None]
        div = jnp.exp(jnp.arange(0, Config.D_MODEL, 2) * -(math.log(10000.0) / Config.D_MODEL))
        pe = jnp.zeros_like(x)
        pe = pe.at[:, :, 0::2].set(jnp.sin(pos * div))
        pe = pe.at[:, :, 1::2].set(jnp.cos(pos * div))
        x = x + pe

        # Scanned Neural Field
        ScannedNF = nn.remat(NeuralFieldLayer)
        
        # We pass 'x' as carry. The second argument to scan is the input sequence.
        # Since we don't have a sequence of inputs per layer, we pass None.
        x, _ = nn.scan(
            ScannedNF,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            length=Config.LAYERS
        )(config=Config)(x, None)

        brain_state = x.astype(jnp.float32)
        
        half = Config.D_MODEL // 2
        u_smith, v_smith, gamma = smith_transform(brain_state[..., :half], brain_state[..., half:])
        logits = nn.Dense(self.vocab_size)(x)
        stats = jnp.stack([u_smith.mean(), v_smith.mean(), gamma.mean()])
        
        return logits, brain_state, stats

# ==============================================================================
# 4. TRAINING ENGINE
# ==============================================================================
def train_process(queue, stop_event):
    try:
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "The neural field is a continuous approximation of discrete neurons. " * 1000
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=Config.VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        data = jnp.array(tokenizer.encode(text).ids)
        
        key = jax.random.PRNGKey(137) 
        model = WubuNeuralField(vocab_size=Config.VOCAB_SIZE)
        dummy_x = jnp.ones((1, Config.SEQ_LEN), dtype=jnp.int32)
        params = model.init(key, dummy_x)['params']
        
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(Config.LR))
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        @jit
        def train_step(state, batch):
            def loss_fn(params):
                logits, brain_state, stats = state.apply_fn({'params': params}, batch)
                
                targets = jnp.roll(batch, -1, axis=1)
                one_hot = jax.nn.one_hot(targets, Config.VOCAB_SIZE)
                loss_ce = optax.softmax_cross_entropy(logits, one_hot).mean()
                
                gamma = stats[2]
                loss_imp = (gamma ** 2) * Config.GAMMA_PENALTY
                
                avg_act = jnp.mean(jnp.abs(brain_state), axis=(0,1))
                diff_sq = (avg_act[:, None] - avg_act[None, :]) ** 2
                loss_geo = jnp.sum(SPATIAL_KERNEL_GPU * diff_sq) / (Config.NUM_NEURONS)
                
                total_loss = loss_ce + loss_imp + (loss_geo * 0.1)
                return total_loss, (loss_ce, loss_geo, gamma, avg_act, stats)
                
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(state.params)
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
            state, loss, (ce, geo, gamma, activity, smith) = train_step(state, batch)
            
            ptr += Config.SEQ_LEN
            step += 1
            
            if step % 2 == 0:
                txt = None
                if step % 20 == 0:
                    try:
                        seed = jnp.array(tokenizer.encode("Neural field").ids).reshape(1, -1)
                        pad = jnp.zeros((1, Config.SEQ_LEN - seed.shape[1]), dtype=jnp.int32)
                        seed_p = jnp.concatenate([pad, seed], axis=1)
                        out = generate(state.params, seed_p)
                        txt = tokenizer.decode(np.array(out[0]))[-100:].replace('\n', ' ')
                    except: pass
                
                packet = {
                    'step': step,
                    'loss': float(loss),
                    'gamma': float(gamma),
                    'smith': np.array(smith),
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
# 5. VISUALIZATION
# ==============================================================================
class WubuFieldGUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1), Layout(name="B", size=4))
        self.layout["M"].split_row(Layout(name="L", ratio=1), Layout(name="R", ratio=1))
        self.angle = 0.0
        self.last_txt = "Initializing Field..."
        self.smith_history = []

    def draw_smith(self, draw, w, h, uv):
        cx, cy = w//2, h//2
        rad = min(w, h) * 0.4
        if rad < 1: return
        
        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], outline=(50,50,50))
        draw.line([cx-rad, cy, cx+rad, cy], fill=(50,50,50))
        
        self.smith_history.append(uv)
        if len(self.smith_history) > 100: self.smith_history.pop(0)
        
        for i, (u, v) in enumerate(self.smith_history):
            px = cx + u * rad
            py = cy - v * rad
            dist = math.sqrt(u**2 + v**2)
            r = int(dist * 255)
            g = int((1-dist) * 255)
            draw.rectangle([px-1, py-1, px+1, py+1], fill=(r, g, 50))

    def render(self, d):
        term_w, term_h = shutil.get_terminal_size()
        w = max(20, (term_w // 2) - 4)
        h = max(20, (term_h - 8) * 2)
        
        try:
            img_3d = Image.new('RGB', (w, h), (5, 5, 10))
            draw_3d = ImageDraw.Draw(img_3d)
            cx, cy = w/2, h/2
            scale = min(w, h) * 0.35
            self.angle += 0.02
            ca, sa = math.cos(self.angle), math.sin(self.angle)
            
            act = d['activity']
            norm = (act - act.min()) / (act.max() - act.min() + 1e-6)
            
            proj = []
            for i, (x, y, z) in enumerate(d['points']):
                rx = x*ca - z*sa
                rz = x*sa + z*ca
                proj.append({'x': cx+rx*scale, 'y': cy+y*scale, 'z': rz, 'v': norm[i], 'i': i})
                
            pmap = {i:p for i,p in enumerate(proj)}
            for i1, i2 in d['wires'][::2]: 
                p1, p2 = pmap[i1], pmap[i2]
                if p1['z'] > -0.5 and p2['z'] > -0.5:
                    v = (p1['v'] + p2['v']) / 2
                    if v > 0.3:
                        draw_3d.line([(p1['x'], p1['y']), (p2['x'], p2['y'])], fill=(0, int(v*255), int(v*255)))

            proj.sort(key=lambda k: k['z'])
            for p in proj:
                if p['z'] > -0.8:
                    sz = max(2, int(scale * 0.02))
                    val = int(p['v'] * 255)
                    col = (val, 255-val, 255) if p['i'] < Config.NUM_NEURONS//2 else (255, 255-val, val)
                    draw_3d.ellipse([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=col)

            img_smith = Image.new('RGB', (w, h), (0, 0, 0))
            draw_smith = ImageDraw.Draw(img_smith)
            self.draw_smith(draw_smith, w, h, (d['smith'][0], d['smith'][1]))

            if d.get('text'): self.last_txt = d['text']
            
            head = f"WUBU NEURAL FIELD | REFLECTION: {d['gamma']:.3f} | LOSS: {d['loss']:.3f}"
            self.layout["T"].update(Panel(head, style="bold white on blue"))
            self.layout["L"].update(Panel(Pixels.from_image(img_3d), title="FIBONACCI LATTICE (2048)"))
            self.layout["R"].update(Panel(Pixels.from_image(img_smith), title="SMITH CHART"))
            self.layout["B"].update(Panel(f"[yellow]{self.last_txt}[/]", title="STREAM"))
            
        except Exception as e:
            self.layout["T"].update(Panel(f"RENDER ERROR: {e}", style="red"))
            
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(); stop = mp.Event()
    console = Console()
    p = mp.Process(target=train_process, args=(q, stop))
    p.start()
    
    gui = WubuFieldGUI()
    ready = False
    
    try:
        with console.status("Generating Fibonacci Lattice...") as status:
            while not ready:
                if not p.is_alive(): raise RuntimeError("Process died")
                try:
                    m = q.get(timeout=0.1)
                    if m.get('type') == 'READY': ready = True
                    if m.get('type') == 'ERROR': raise RuntimeError(m['msg'])
                except mp.queues.Empty: pass

        with Live(gui.layout, refresh_per_second=12, screen=True) as live:
            while True:
                if not p.is_alive(): raise RuntimeError("Process died")
                try:
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA': live.update(gui.render(msg['data']))
                    elif msg['type'] == 'ERROR': raise RuntimeError(msg['msg'])
                except mp.queues.Empty: time.sleep(0.01)

    except KeyboardInterrupt: stop.set()
    except Exception as e:
        stop.set()
        console.clear()
        console.print(Panel(f"[bold red]CRASH REPORT[/]\n{e}", border_style="red"))
        while not q.empty():
            m = q.get()
            if m.get('type') == 'ERROR': console.print(f"[dim]{m['msg']}[/dim]")
    finally:
        if p.is_alive(): p.terminate(); p.join()