# Wubu_VGA_Ultimate.py
# THE GEOMETRIC BRAIN [VGA CANVAS ARCHITECTURE]
#
# FEATURE: Persistence (Save/Load) + Text Generation + layout V2.
# OPTIMIZATION: Non-blocking Async Queues + JIT Compilation.

import multiprocessing as mp
import time
import requests
import numpy as np
import os
import math
import shutil
import traceback
import pickle
from typing import NamedTuple

# GUI
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich_pixels import Pixels
from PIL import Image, ImageDraw

# ==============================================================================
# 1. THE GEODESIC ENGINE (GPU + LOGIC)
# ==============================================================================
class GeoConfig:
    SEQ_LEN = 1024          
    BATCH_SIZE = 1          
    NUM_NEURONS = 1024      
    NEIGHBORS = 6           
    LR = 0.002
    SAVE_EVERY = 1000
    GEN_EVERY = 20          # Generate text every N steps (for speed)
    SAVE_FILE = "wubu_core.pkl"

def island_process(queue, stop_event):
    # Disable preallocation to play nice with desktop
    os.environ['JAX_PLATFORMS'] = '' 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    try:
        import jax
        import jax.numpy as jnp
        import optax
        
        # 64-bit precision for stable gradients on recursive quaternions
        jax.config.update("jax_enable_x64", False) 
        
        # --- 1.1 GEOMETRY SETUP ---
        def generate_sphere_points(n):
            points = []
            phi = math.pi * (3. - math.sqrt(5.))
            for i in range(n):
                y = 1 - (i / float(n - 1)) * 2 
                radius = math.sqrt(1 - y * y)
                theta = phi * i 
                x = math.cos(theta) * radius
                z = math.sin(theta) * radius
                points.append((x, y, z))
            return np.array(points, dtype=np.float32)

        points_np = generate_sphere_points(GeoConfig.NUM_NEURONS)
        dists = np.linalg.norm(points_np[:, None] - points_np[None, :], axis=-1)
        
        # Create adjacency mask (GPU constant)
        mask = np.zeros((GeoConfig.NUM_NEURONS, GeoConfig.NUM_NEURONS), dtype=np.float32)
        connections = []
        for i in range(GeoConfig.NUM_NEURONS):
            nearest = np.argsort(dists[i])[:GeoConfig.NEIGHBORS+1]
            mask[i, nearest] = 1.0
            for t in nearest:
                if i < t: connections.append((i, t))
            
        MASK_GPU = jax.device_put(jnp.array(mask))

        # --- 1.2 QUATERNION MATH ---
        def split_quat(tensor): return jnp.split(tensor, 4, axis=-1)
        def merge_quat(r, i, j, k): return jnp.concatenate([r, i, j, k], axis=-1)

        def hamilton_product(r, i, j, k, w_r, w_i, w_j, w_k):
            o_r = jnp.dot(r, w_r) - jnp.dot(i, w_i) - jnp.dot(j, w_j) - jnp.dot(k, w_k)
            o_i = jnp.dot(r, w_i) + jnp.dot(i, w_r) + jnp.dot(j, w_k) - jnp.dot(k, w_j)
            o_j = jnp.dot(r, w_j) - jnp.dot(i, w_k) + jnp.dot(j, w_r) + jnp.dot(k, w_i)
            o_k = jnp.dot(r, w_k) + jnp.dot(i, w_j) - jnp.dot(j, w_i) + jnp.dot(k, w_r)
            return o_r, o_i, o_j, o_k

        def init_params(key, vocab, n_neurons):
            k1, k2, k3 = jax.random.split(key, 3)
            # Recurrent weights initialized near Identity for long-term memory
            w_r = jnp.eye(n_neurons) * 0.9 + jax.random.normal(k1, (n_neurons, n_neurons)) * 0.01
            w_i = jax.random.normal(k2, (n_neurons, n_neurons)) * 0.01
            w_j = jax.random.normal(k2, (n_neurons, n_neurons)) * 0.01
            w_k = jax.random.normal(k3, (n_neurons, n_neurons)) * 0.01
            return {
                'q_rec': {'r': w_r, 'i': w_i, 'j': w_j, 'k': w_k},
                'w_emb': jax.random.normal(k1, (vocab, n_neurons * 4)) * 0.02,
                'w_out': jax.random.normal(k3, (n_neurons * 4, vocab)) * 0.02,
                'b_rec': jnp.zeros((n_neurons * 4,)),
                'b_out': jnp.zeros((vocab,))
            }

        # --- 1.3 CORE LOGIC ---
        def forward(params, x_seq, h_init):
            # Apply geometric mask to weights
            wr = params['q_rec']['r'] * MASK_GPU
            wi = params['q_rec']['i'] * MASK_GPU
            wj = params['q_rec']['j'] * MASK_GPU
            wk = params['q_rec']['k'] * MASK_GPU
            
            def step(h, x_idx):
                x = params['w_emb'][x_idx]
                h_r, h_i, h_j, h_k = split_quat(h)
                x_r, x_i, x_j, x_k = split_quat(x)
                
                # Quaternion Rotation (The "Thinking")
                rot_r, rot_i, rot_j, rot_k = hamilton_product(h_r, h_i, h_j, h_k, wr, wi, wj, wk)
                b_r, b_i, b_j, b_k = split_quat(params['b_rec'])
                
                # Nonlinearity
                next_r = jnp.tanh(rot_r + x_r + b_r)
                next_i = jnp.tanh(rot_i + x_i + b_i)
                next_j = jnp.tanh(rot_j + x_j + b_j)
                next_k = jnp.tanh(rot_k + x_k + b_k)
                
                h_new = merge_quat(next_r, next_i, next_j, next_k)
                logits = jnp.dot(h_new, params['w_out']) + params['b_out']
                return h_new, logits

            def run_row(h_prev, row_x): return jax.lax.scan(step, h_prev, row_x)
            h_final, logits_seq = jax.vmap(run_row)(h_init, x_seq)
            return h_final, logits_seq

        @jax.jit
        def train_step(params, opt_state, h_state, x_batch, y_batch):
            def loss_fn(p):
                _, logits = forward(p, x_batch, h_state)
                # Flatten batch/seq for loss
                logits_flat = logits.reshape(-1, logits.shape[-1])
                y_flat = y_batch.reshape(-1)
                one_hot = jax.nn.one_hot(y_flat, 65)
                log_probs = jax.nn.log_softmax(logits_flat)
                return -jnp.sum(one_hot * log_probs) / y_flat.size

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = optax.adam(GeoConfig.LR).update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            # Reset hidden state occasionally or keep it? For now, reset to avoid gradient explosion in long runs
            return new_params, new_opt, jnp.zeros_like(h_state), loss

        @jax.jit
        def predict_step(params, h, char_idx):
            # Single step prediction for text generation
            # Manually unrolled step from forward()
            wr = params['q_rec']['r'] * MASK_GPU
            wi = params['q_rec']['i'] * MASK_GPU
            wj = params['q_rec']['j'] * MASK_GPU
            wk = params['q_rec']['k'] * MASK_GPU
            
            x = params['w_emb'][char_idx]
            h_r, h_i, h_j, h_k = split_quat(h)
            x_r, x_i, x_j, x_k = split_quat(x)
            
            rot_r, rot_i, rot_j, rot_k = hamilton_product(h_r, h_i, h_j, h_k, wr, wi, wj, wk)
            b_r, b_i, b_j, b_k = split_quat(params['b_rec'])
            
            next_r = jnp.tanh(rot_r + x_r + b_r)
            next_i = jnp.tanh(rot_i + x_i + b_i)
            next_j = jnp.tanh(rot_j + x_j + b_j)
            next_k = jnp.tanh(rot_k + x_k + b_k)
            
            h_new = merge_quat(next_r, next_i, next_j, next_k)
            logits = jnp.dot(h_new, params['w_out']) + params['b_out']
            return h_new, logits

        # --- 1.4 DATA LOADING & INIT ---
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Wubu is life. " * 10000
        chars = sorted(list(set(text)))
        char_to_ix = {ch:i for i,ch in enumerate(chars)}
        ix_to_char = {i:ch for i,ch in enumerate(chars)}
        raw_data = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)
        data_gpu = jax.device_put(jnp.array(raw_data))

        # STATE INITIALIZATION
        key = jax.random.PRNGKey(42)
        
        # Checkpoint Loading
        start_step = 0
        if os.path.exists(GeoConfig.SAVE_FILE):
            try:
                with open(GeoConfig.SAVE_FILE, 'rb') as f:
                    saved = pickle.load(f)
                params = saved['params']
                opt_state = saved['opt_state']
                start_step = saved['step']
                # Re-put to device
                params = jax.tree_util.tree_map(jax.device_put, params)
                # opt_state structure is complex, usually fine to load as is if optax version matches
                queue.put({'type': 'LOG', 'msg': f"RESUMED FROM STEP {start_step}"})
            except Exception as e:
                queue.put({'type': 'LOG', 'msg': f"LOAD FAILED: {e}"})
                params = init_params(key, len(chars), GeoConfig.NUM_NEURONS)
                opt_state = optax.adam(GeoConfig.LR).init(params)
        else:
            params = init_params(key, len(chars), GeoConfig.NUM_NEURONS)
            opt_state = optax.adam(GeoConfig.LR).init(params)

        h_state = jnp.zeros((GeoConfig.BATCH_SIZE, GeoConfig.NUM_NEURONS * 4))

        # Trigger JIT compilation
        dummy = jnp.zeros((1, GeoConfig.SEQ_LEN), dtype=jnp.int32)
        train_step(params, opt_state, h_state, dummy, dummy)
        queue.put({'type': 'READY'})

        # --- 1.5 TRAINING LOOP ---
        ptr = (start_step * GeoConfig.SEQ_LEN) % len(raw_data)
        current_text = "Initializing..."
        
        step_count = start_step
        
        while not stop_event.is_set():
            if ptr + GeoConfig.SEQ_LEN + 1 > len(raw_data): ptr = 0
            
            x_batch = jax.lax.dynamic_slice(data_gpu, (ptr,), (GeoConfig.SEQ_LEN,)).reshape(1, GeoConfig.SEQ_LEN)
            y_batch = jax.lax.dynamic_slice(data_gpu, (ptr+1,), (GeoConfig.SEQ_LEN,)).reshape(1, GeoConfig.SEQ_LEN)
            
            params, opt_state, h_state, loss = train_step(params, opt_state, h_state, x_batch, y_batch)
            ptr += GeoConfig.SEQ_LEN
            step_count += 1
            
            # --- TEXT GENERATION (Periodic) ---
            if step_count % GeoConfig.GEN_EVERY == 0:
                # Seed with "Wubu is"
                seed_str = "Wubu is"
                # If char not in vocab, skip it
                idxs = [char_to_ix[c] for c in seed_str if c in char_to_ix]
                
                # Run seed
                gen_h = jnp.zeros((GeoConfig.NUM_NEURONS * 4,)) # Reset hidden for gen
                for ix in idxs:
                    gen_h, _ = predict_step(params, gen_h, ix)
                
                # Generate new chars
                generated = seed_str
                next_ix = idxs[-1]
                for _ in range(120): # Length of footer
                    gen_h, logits = predict_step(params, gen_h, next_ix)
                    # Greedy sampling (fastest for gaming loop)
                    next_ix = int(jnp.argmax(logits))
                    generated += ix_to_char[next_ix]
                    if next_ix == char_to_ix.get('\n', -1): break
                current_text = generated.replace('\n', ' ')

            # --- CHECKPOINTING (Periodic) ---
            if step_count % GeoConfig.SAVE_EVERY == 0:
                # Move to CPU for pickling
                save_data = {
                    'params': jax.tree_util.tree_map(lambda x: np.array(x), params),
                    'opt_state': optax.adam(GeoConfig.LR).init(jax.tree_util.tree_map(lambda x: np.array(x), params)), # Simplified save
                    'step': step_count
                }
                try:
                    with open(GeoConfig.SAVE_FILE, 'wb') as f:
                        pickle.dump(save_data, f)
                except: pass

            # --- VISUALIZATION DATA ---
            # Send data every frame (or every N frames)
            # Use put_nowait to ensure Physics doesn't wait for Render
            try:
                # Extract Diagonal Activity (Lightweight)
                r = np.array(jnp.diag(params['q_rec']['r']))
                i = np.array(jnp.diag(params['q_rec']['i']))
                j = np.array(jnp.diag(params['q_rec']['j']))
                k = np.array(jnp.diag(params['q_rec']['k']))
                
                packet = {
                    'loss': float(loss), 
                    'step': step_count,
                    'q': {'r': r, 'i': i, 'j': j, 'k': k}, 
                    'points': points_np, 
                    'wires': connections,
                    'text': current_text
                }
                queue.put_nowait({'type': 'DATA', 'data': packet})
            except mp.queues.Full:
                pass # Skip frame if GUI is slow

    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 2. VGA CANVAS ENGINE
# ==============================================================================
class VGABrain:
    def __init__(self):
        # NEW LAYOUT: Header, Main (Split L/R), Footer (Text)
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="T", size=3),
            Layout(name="M", ratio=1),
            Layout(name="B", size=3)
        )
        self.layout["M"].split_row(
            Layout(name="L", ratio=1), 
            Layout(name="R", ratio=1)
        )
        
        self.angle = 0.0
        self.last_data = None

    def get_panel_dims(self):
        term_w, term_h = shutil.get_terminal_size()
        # Header=3, Footer=3, Padding=2. Total Vertical Deduction = 8
        w = max(10, (term_w // 2) - 4)
        h = max(10, (term_h - 8) * 2) 
        return w, h

    def render_sphere(self, d, w, h):
        img = Image.new('RGB', (w, h), (5, 5, 15)) 
        draw = ImageDraw.Draw(img)
        cx, cy = w / 2, h / 2
        min_dim = min(w, h)
        
        SCALE = min_dim * 0.40          
        NODE_SZ = max(1, min_dim * 0.008)
        
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        
        points = d['points']
        r_vals = d['q']['r']
        v_min, v_max = r_vals.min(), r_vals.max()
        
        proj = []
        for idx, (x, y, z) in enumerate(points):
            rx = x * cos_a - z * sin_a
            ry = y
            rz = x * sin_a + z * cos_a
            px, py = cx + (rx * SCALE), cy + (ry * SCALE)
            v = (r_vals[idx] - v_min) / (v_max - v_min + 1e-6)
            proj.append({'x': px, 'y': py, 'z': rz, 'v': v, 'idx': idx})
            
        # Wires (Back only)
        pmap = {p['idx']: p for p in proj}
        for i1, i2 in d['wires']:
            p1, p2 = pmap[i1], pmap[i2]
            if p1['z'] > -0.5 and p2['z'] > -0.5: 
                depth = (p1['z'] + p2['z']) / 2
                b = int((depth + 1.0) * 100) + 50
                # Optimize: only draw bright wires
                if b > 100:
                    draw.line([(p1['x'], p1['y']), (p2['x'], p2['y'])], fill=(0, b, b))
                
        # Nodes
        proj.sort(key=lambda k: k['z'])
        for p in proj:
            if p['z'] > 0: 
                z_mult = 1.0 if p['z'] > 0.5 else 0.7
                sz = max(1, int(NODE_SZ * z_mult))
                val = int(p['v'] * 255)
                # Cyan/Magenta/White scheme
                col = (val, 255-val, 255)
                draw.ellipse([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=col)
        return img

    def render_lattice(self, d, w, h):
        img = Image.new('RGB', (w, h), (10, 10, 10)) 
        draw = ImageDraw.Draw(img)
        COLS = 32
        ROWS = 32
        
        r = d['q']['r']
        i_v, j_v, k_v = d['q']['i'], d['q']['j'], d['q']['k']
        
        v_min, v_max = r.min(), r.max()
        norm_r = (r - v_min) / (v_max - v_min + 1e-6)
        
        cell_w = w / COLS
        PAD = 1 if cell_w > 8 else 0
        
        for idx in range(1024):
            c, row = idx % COLS, idx // COLS
            x0, y0 = int(c * w / COLS), int(row * h / ROWS)
            x1, y1 = int((c + 1) * w / COLS), int((row + 1) * h / ROWS)
            
            bright = 0.5 + norm_r[idx] * 0.5
            red = int(min(255, (i_v[idx]*10+0.5)*255) * bright)
            grn = int(min(255, (j_v[idx]*10+0.5)*255) * bright)
            blu = int(min(255, (k_v[idx]*10+0.5)*255) * bright)
            
            draw.rectangle([x0+PAD, y0+PAD, x1-PAD, y1-PAD], fill=(red, grn, blu))
        return img

    def update(self, d):
        if d: self.last_data = d
        if not self.last_data: return self.layout
        
        self.angle += 0.05 # Slightly faster rotation
        
        # Header
        head = f"[bold green]WUBU VGA CORE[/bold green] | STEP: {self.last_data['step']} | LOSS: {self.last_data['loss']:.4f}"
        self.layout["T"].update(Panel(head, style="white on black"))
        
        # Main
        w, h = self.get_panel_dims()
        img_3d = self.render_sphere(self.last_data, w, h)
        img_2d = self.render_lattice(self.last_data, w, h)
        
        self.layout["L"].update(Panel(Pixels.from_image(img_3d), title="SPHERE (Dynamic)", style="white on black"))
        self.layout["R"].update(Panel(Pixels.from_image(img_2d), title="LATTICE (1024 Units)", style="white on black"))
        
        # Footer (Text Output)
        txt = self.last_data.get('text', 'Waiting for thoughts...')
        self.layout["B"].update(Panel(Align.center(f"[bold cyan]{txt}[/bold cyan]"), title="SUBCONSCIOUS STREAM", style="white on black"))
        
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=2); stop = mp.Event() # maxsize 2 gives a tiny buffer
    console = Console(); console.clear()
    
    p = mp.Process(target=island_process, args=(q, stop))
    p.start()
    
    gui = VGABrain(); ready = False
    
    with console.status("[bold green]Booting VGA Interface & JIT Compiling..."):
        while not ready:
            try:
                m = q.get()
                if m.get('type') == 'READY': ready = True
                if m.get('type') == 'LOG': console.print(f"[dim]{m['msg']}[/dim]")
            except: pass
            
    try:
        # refresh_per_second=20 is the sweet spot for terminal performance
        with Live(gui.layout, refresh_per_second=20, screen=True) as live:
            while True:
                try:
                    # Non-blocking get with a tiny sleep to let Physics breathe
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA': 
                        live.update(gui.update(msg['data']))
                    elif msg['type'] == 'ERROR':
                        stop.set()
                        print(msg['msg'])
                        break
                except mp.queues.Empty:
                    # Animate rotation even if no new data came in
                    live.update(gui.update(None))
                    time.sleep(0.01)
    except KeyboardInterrupt:
        stop.set()
    finally:
        p.join()