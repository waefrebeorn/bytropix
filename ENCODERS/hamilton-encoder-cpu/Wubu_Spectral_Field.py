# Wubu_Spectral_Field.py
# THE GEOMETRIC BRAIN v7.5 [FINAL STABILITY]
#
# ARCHITECTURE: Geodesic Transformer (Spectral Graph Wave Equation).
# FIX: Solved Unpacking Error (5 vs 6 vars).
# FIX: Solved Double-Execution of Manifold Generation.

import os
import sys
import time
import math
import traceback
import multiprocessing as mp
import requests
import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import shutil
from typing import Any, Tuple, List

# --- JAX CONFIG ---
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
    NEIGHBORS = 12          
    NORMAL_THRESH = 0.6     
    
    # SPECTRAL DYNAMICS
    DT = 0.2                
    DAMPING = 0.1           
    PROP_SPEED = 2.0        
    
    # TRANSFORMER
    SEQ_LEN = 128
    D_MODEL = 512           
    LAYERS = 2
    
    # LEARNING
    LR = 3e-4
    VOCAB_SIZE = 8192

# ==============================================================================
# 2. GEODESIC MANIFOLD CONSTRUCTION
# ==============================================================================
def generate_spectral_brain():
    print("Constructing Riemannian Manifold...", file=sys.stderr)
    
    def map_brain(p):
        d_base = np.linalg.norm(p / np.array([1.0, 0.8, 1.2]), axis=1) - 1.0
        gyri = 0.05 * np.sin(10*p[:,0]) * np.sin(10*p[:,1]) * np.sin(10*p[:,2])
        d_fissure = np.abs(p[:,0]) - 0.03
        return np.maximum(d_base + gyri, -d_fissure)

    def get_normals(p, delta=0.01):
        n = np.zeros_like(p)
        for i in range(3):
            p1 = p.copy(); p1[:,i] += delta
            p2 = p.copy(); p2[:,i] -= delta
            n[:,i] = map_brain(p1) - map_brain(p2)
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

    points = []
    normals = []
    batch_size = 50000
    attempts = 0
    
    while len(points) < Config.NUM_NEURONS and attempts < 100:
        p_rnd = (np.random.rand(batch_size, 3) * 3.0) - 1.5
        d_val = map_brain(p_rnd)
        mask = np.abs(d_val) < 0.03 
        valid_p = p_rnd[mask]
        
        if len(valid_p) > 0:
            valid_n = get_normals(valid_p)
            for i in range(len(valid_p)):
                points.append(valid_p[i])
                normals.append(valid_n[i])
                if len(points) >= Config.NUM_NEURONS: break
        attempts += 1
            
    points = np.array(points[:Config.NUM_NEURONS], dtype=np.float32)
    normals = np.array(normals[:Config.NUM_NEURONS], dtype=np.float32)
    
    print("Building Geodesic Laplacian...", file=sys.stderr)
    nbrs = NearestNeighbors(n_neighbors=Config.NEIGHBORS * 2).fit(points)
    dist, idx = nbrs.kneighbors(points)
    
    adj_rows = []
    adj_cols = []
    adj_data = []
    wires = []
    
    for i in range(Config.NUM_NEURONS):
        valid_neighbors = 0
        for j_idx, d in zip(idx[i], dist[i]):
            if i == j_idx: continue
            dot = np.dot(normals[i], normals[j_idx])
            
            if dot > Config.NORMAL_THRESH:
                weight = np.exp(-d * 4.0) 
                adj_rows.append(i)
                adj_cols.append(j_idx)
                adj_data.append(weight)
                if valid_neighbors < 2: 
                    if i < j_idx: wires.append((i, j_idx))
                valid_neighbors += 1
            if valid_neighbors >= Config.NEIGHBORS: break

    N = Config.NUM_NEURONS
    adj_mat = scipy.sparse.coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(N, N))
    adj_mat = (adj_mat + adj_mat.T) / 2.0 
    
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
    d_mat_inv = scipy.sparse.diags(d_inv_sqrt)
    laplacian = scipy.sparse.eye(N) - d_mat_inv @ adj_mat @ d_mat_inv
    L_dense = jnp.array(laplacian.toarray())
    
    mask_in = (points[:, 0] < -0.1) & (points[:, 2] < -0.2)
    mask_out = (points[:, 0] < -0.1) & (points[:, 2] > 0.2)
    
    # Return exactly 5 items
    return points, wires, L_dense, jnp.array(mask_in, dtype=float), jnp.array(mask_out, dtype=float)

# --- GLOBAL EXECUTION (Run once per process) ---
# This ensures JAX can see the Laplacian as a global constant for JIT compilation.
POINTS_3D, WIRES, LAPLACIAN_GPU, MASK_IN, MASK_OUT = generate_spectral_brain()

# ==============================================================================
# 3. SPECTRAL DYNAMICS
# ==============================================================================

class SpectralWaveLayer(nn.Module):
    config: Config
    
    @nn.compact
    def __call__(self, state, forcing_input):
        u, v = state 
        # Global Laplacian (Constant)
        elastic_force = -Config.PROP_SPEED * jnp.einsum('nm,bmc->bnc', LAPLACIAN_GPU, u)
        friction_force = -Config.DAMPING * v
        
        # Learnable Drive (RNN Weights)
        drive = nn.Dense(u.shape[-1])(forcing_input)
        
        total_force = elastic_force + friction_force + drive
        
        v_new = v + total_force * Config.DT
        u_new = u + v_new * Config.DT
        u_new = jnp.tanh(u_new)
        
        return (u_new, v_new), u_new

class GeodesicTransformer(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, x, train=True):
        B, T = x.shape
        x_emb = nn.Embed(self.vocab_size, Config.D_MODEL)(x)
        sensory = nn.Dense(Config.NUM_NEURONS)(x_emb)
        sensory = sensory * MASK_IN[None, None, :]
        
        # Input: [B, T, N, 1]
        scan_input = sensory[..., None]
        
        # RNN Scan
        ScanRNN = nn.scan(
            nn.remat(SpectralWaveLayer),
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )
        
        init_u = jnp.zeros((B, Config.NUM_NEURONS, 1))
        init_v = jnp.zeros((B, Config.NUM_NEURONS, 1))
        init_state = (init_u, init_v)
        
        final_state, wave_history = ScanRNN(config=Config)(init_state, scan_input)
        
        motor = wave_history.squeeze(-1) * MASK_OUT[None, None, :] 
        motor = motor.transpose(1, 0, 2) # [B, T, N]
        
        logits = nn.Dense(self.vocab_size)(motor)
        avg_energy = jnp.mean(wave_history**2)
        
        return logits, wave_history[:,-1,:,:].squeeze(), avg_energy

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train_process(queue, stop_event):
    try:
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Wubu " * 5000
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=Config.VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        data = jnp.array(tokenizer.encode(text).ids)

        key = jax.random.PRNGKey(42)
        model = GeodesicTransformer(vocab_size=Config.VOCAB_SIZE)
        dummy_x = jnp.zeros((1, Config.SEQ_LEN), dtype=jnp.int32)
        params = model.init(key, dummy_x)['params']
        
        tx = optax.adamw(Config.LR)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        @jit
        def train_step(state, batch):
            def loss_fn(params):
                logits, brain_state, energy = state.apply_fn({'params': params}, batch)
                targets = jnp.roll(batch, -1, axis=1)
                one_hot = jax.nn.one_hot(targets, Config.VOCAB_SIZE)
                loss = optax.softmax_cross_entropy(logits, one_hot).mean()
                return loss + 0.001 * energy, (loss, energy, brain_state)
                
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(state.params)
            return state.apply_gradients(grads=grads), loss, aux

        ptr = 0
        step = 0
        queue.put({'type': 'READY'})
        
        while not stop_event.is_set():
            if ptr + Config.SEQ_LEN + 1 > len(data): ptr = 0
            batch = jax.lax.dynamic_slice(data, (ptr,), (Config.SEQ_LEN,)).reshape(1, Config.SEQ_LEN)
            
            state, loss, (ce, energy, act) = train_step(state, batch)
            ptr += Config.SEQ_LEN
            step += 1
            
            if step % 2 == 0:
                packet = {
                    'step': step,
                    'loss': float(ce),
                    'energy': float(energy),
                    'activity': np.array(act).flatten(),
                    'points': POINTS_3D, 
                    'wires': WIRES,      
                    'mask_in': np.array(MASK_IN),
                    'mask_out': np.array(MASK_OUT)
                }
                try: queue.put_nowait({'type': 'DATA', 'data': packet})
                except: pass

    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 5. ADVANCED TELEMETRY GUI
# ==============================================================================
class SpectralGUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1))
        self.layout["M"].split_row(Layout(name="L", ratio=2), Layout(name="R", ratio=1))
        
        self.angle = 0.0
        self.loss_history = []
        self.energy_history = []

    def draw_graph(self, draw, w, h, data, color, title):
        draw.rectangle([0,0,w,h], fill=(10,10,10))
        draw.rectangle([0,0,w,h], outline=(50,50,50))
        
        if len(data) < 2: return
        
        d_min, d_max = min(data), max(data)
        rng = d_max - d_min + 1e-6
        
        pts = []
        for i, val in enumerate(data):
            x = int((i / (len(data)-1)) * (w-10)) + 5
            y = int(h - 5 - ((val - d_min) / rng) * (h-20))
            pts.append((x,y))
            
        draw.line(pts, fill=color, width=2)
        draw.text((5, 5), f"{title}: {data[-1]:.4f}", fill=(200,200,200))

    def render(self, d):
        term_w, term_h = shutil.get_terminal_size()
        
        self.loss_history.append(d['loss'])
        self.energy_history.append(d['energy'])
        if len(self.loss_history) > 100: self.loss_history.pop(0)
        if len(self.energy_history) > 100: self.energy_history.pop(0)
        
        w_b = int(term_w * 0.66) - 4
        h_b = (term_h - 4) * 2
        
        if w_b < 10 or h_b < 10: return Panel("Resize...", style="red")

        try:
            img_brain = Image.new('RGB', (w_b, h_b), (5, 5, 10))
            draw_b = ImageDraw.Draw(img_brain)
            
            cx, cy = w_b/2, h_b/2
            scale = min(w_b, h_b) * 0.35
            self.angle += 0.02
            ca, sa = math.cos(self.angle), math.sin(self.angle)
            
            act = d['activity']
            act = 1.0 / (1.0 + np.exp(-3.0 * act)) 
            
            proj = []
            for i, (x, y, z) in enumerate(d['points']):
                rx = x*ca - z*sa
                rz = x*sa + z*ca
                
                base_col = (40, 40, 80)
                if d['mask_in'][i]: base_col = (40, 180, 40)
                elif d['mask_out'][i]: base_col = (180, 40, 40)
                
                intensity = act[i]
                r = int(base_col[0] * (1-intensity) + 255 * intensity)
                g = int(base_col[1] * (1-intensity) + 255 * intensity)
                b = int(base_col[2] * (1-intensity) + 255 * intensity)
                
                proj.append({'x': cx+rx*scale, 'y': cy+y*scale, 'z': rz, 'c': (r,g,b)})
                
            pmap = {i:p for i,p in enumerate(proj)}
            for i1, i2 in d['wires']:
                p1, p2 = pmap[i1], pmap[i2]
                if p1['z'] > -0.5 and p2['z'] > -0.5:
                    draw_b.line([(p1['x'], p1['y']), (p2['x'], p2['y'])], fill=(30,30,50))
                    
            proj.sort(key=lambda k: k['z'])
            for p in proj:
                if p['z'] > -0.8:
                    sz = 3 if p['z'] > 0 else 2
                    draw_b.rectangle([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=p['c'])

            w_t = term_w - w_b - 6
            h_t = h_b
            h_graph = h_t // 2
            
            img_tele = Image.new('RGB', (w_t, h_t), (0,0,0))
            draw_t = ImageDraw.Draw(img_tele)
            
            self.draw_graph(draw_t, w_t, h_graph, self.loss_history, (255, 100, 100), "LOSS")
            
            d_min, d_max = min(self.energy_history), max(self.energy_history)
            rng = d_max - d_min + 1e-6
            if len(self.energy_history) > 1:
                pts = []
                for i, val in enumerate(self.energy_history):
                    x = int((i / (len(self.energy_history)-1)) * (w_t-10)) + 5
                    y = int((h_t - 5) - ((val - d_min) / rng) * (h_graph-20))
                    pts.append((x,y))
                draw_t.rectangle([0, h_graph, w_t, h_t], outline=(50,50,50))
                draw_t.line(pts, fill=(100, 255, 255), width=2)
                draw_t.text((5, h_graph+5), f"WAVE ENERGY: {self.energy_history[-1]:.4f}", fill=(200,200,200))

            head_txt = f"WUBU SPECTRAL FIELD | STEP {d['step']} | NEURONS: {Config.NUM_NEURONS}"
            self.layout["T"].update(Panel(head_txt, style="bold white on blue"))
            self.layout["L"].update(Panel(Pixels.from_image(img_brain), title="RIEMANNIAN MANIFOLD"))
            self.layout["R"].update(Panel(Pixels.from_image(img_tele), title="TELEMETRY"))
            
        except Exception as e:
            self.layout["T"].update(Panel(f"RENDER ERROR: {e}", style="red"))
            
        return self.layout

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except: pass

    q = mp.Queue(); stop = mp.Event()
    console = Console()
    
    # FIX: Geometry is ALREADY generated at Module Level (lines 135-136).
    # We do NOT run it here to avoid the 6-value vs 5-value unpacking error.
    # The Child Process will simply import the module and get the same Globals.
    
    p = mp.Process(target=train_process, args=(q, stop))
    p.start()
    
    gui = SpectralGUI()
    ready = False
    
    try:
        # Phase 1: Wait for READY
        with console.status("Initializing Quantum Dynamics...") as status:
            while not ready:
                if not p.is_alive():
                    raise RuntimeError("Process died during initialization")
                try:
                    m = q.get(timeout=0.1)
                    if m.get('type') == 'READY': ready = True
                    if m.get('type') == 'ERROR': raise RuntimeError(m['msg'])
                except mp.queues.Empty: pass

        # Phase 2: Live Loop
        with Live(gui.layout, refresh_per_second=12, screen=True) as live:
            while True:
                if not p.is_alive():
                    raise RuntimeError("Process died unexpectedly")
                try:
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA': 
                        live.update(gui.render(msg['data']))
                    elif msg['type'] == 'ERROR':
                        raise RuntimeError(msg['msg'])
                except mp.queues.Empty: time.sleep(0.01)

    except KeyboardInterrupt:
        stop.set()
    except Exception as e:
        stop.set()
        console.clear()
        console.print(Panel(f"[bold red]FATAL CRASH REPORT[/bold red]\n\n{str(e)}", border_style="red"))
        while not q.empty():
            m = q.get()
            if m.get('type') == 'ERROR':
                console.print(f"\n[bold yellow]Detailed Traceback:[/bold yellow]\n{m['msg']}")
    finally:
        if p.is_alive(): p.terminate()