# Wubu_VGA_Canvas.py
# THE GEOMETRIC BRAIN [VGA CANVAS ARCHITECTURE]
#
# LOGIC: Dynamic Resolution -> Matches Terminal Split Exactly.
# 3D: Orthographic Projection (Dynamic Element Scaling).
# 2D: Integer-snapped Grid (Clean borders).

import multiprocessing as mp
import time
import requests
import numpy as np
import os
import math
import shutil
import traceback
from typing import NamedTuple

# GUI
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich_pixels import Pixels
from PIL import Image, ImageDraw

# ==============================================================================
# 1. THE GEODESIC ENGINE (GPU)
# ==============================================================================
class GeoConfig:
    SEQ_LEN = 1024          
    BATCH_SIZE = 1          
    NUM_NEURONS = 1024      # 32x32 Grid
    NEIGHBORS = 6           
    LR = 0.002              

def island_process(queue, stop_event):
    os.environ['JAX_PLATFORMS'] = '' 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    try:
        import jax
        import jax.numpy as jnp
        import optax
        
        jax.config.update("jax_enable_x64", False) 
        
        # --- GEOMETRY ---
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
        
        mask = np.zeros((GeoConfig.NUM_NEURONS, GeoConfig.NUM_NEURONS), dtype=np.float32)
        connections = []
        for i in range(GeoConfig.NUM_NEURONS):
            nearest = np.argsort(dists[i])[:GeoConfig.NEIGHBORS+1]
            mask[i, nearest] = 1.0
            for t in nearest:
                if i < t: connections.append((i, t))
            
        MASK_GPU = jax.device_put(jnp.array(mask))

        # --- MATH ---
        def hamilton_product(r, i, j, k, w_r, w_i, w_j, w_k):
            o_r = jnp.dot(r, w_r) - jnp.dot(i, w_i) - jnp.dot(j, w_j) - jnp.dot(k, w_k)
            o_i = jnp.dot(r, w_i) + jnp.dot(i, w_r) + jnp.dot(j, w_k) - jnp.dot(k, w_j)
            o_j = jnp.dot(r, w_j) - jnp.dot(i, w_k) + jnp.dot(j, w_r) + jnp.dot(k, w_i)
            o_k = jnp.dot(r, w_k) + jnp.dot(i, w_j) - jnp.dot(j, w_i) + jnp.dot(k, w_r)
            return o_r, o_i, o_j, o_k

        def split_quat(tensor): return jnp.split(tensor, 4, axis=-1)
        def merge_quat(r, i, j, k): return jnp.concatenate([r, i, j, k], axis=-1)

        def init_params(key, vocab, n_neurons):
            k1, k2, k3 = jax.random.split(key, 3)
            w_r = jnp.eye(n_neurons) * 0.8 + jax.random.normal(k1, (n_neurons, n_neurons)) * 0.01
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

        def forward(params, x_seq, h_init):
            wr = params['q_rec']['r'] * MASK_GPU
            wi = params['q_rec']['i'] * MASK_GPU
            wj = params['q_rec']['j'] * MASK_GPU
            wk = params['q_rec']['k'] * MASK_GPU
            
            def step(h, x_idx):
                x = params['w_emb'][x_idx]
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

            def run_row(h_prev, row_x): return jax.lax.scan(step, h_prev, row_x)
            h_final, logits_seq = jax.vmap(run_row)(h_init, x_seq)
            return h_final, logits_seq

        @jax.jit
        def train_step(params, opt_state, h_state, x_batch, y_batch):
            def loss_fn(p):
                _, logits = forward(p, x_batch, h_state)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                y_flat = y_batch.reshape(-1)
                one_hot = jax.nn.one_hot(y_flat, 65)
                log_probs = jax.nn.log_softmax(logits_flat)
                return -jnp.sum(one_hot * log_probs) / y_flat.size

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = optax.adam(GeoConfig.LR).update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt, jnp.zeros_like(h_state), loss

        # --- RUN LOOP ---
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Fail " * 200000
        chars = sorted(list(set(text)))
        char_to_ix = {ch:i for i,ch in enumerate(chars)}
        raw_data = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)
        data_gpu = jax.device_put(jnp.array(raw_data))

        key = jax.random.PRNGKey(42)
        params = init_params(key, len(chars), GeoConfig.NUM_NEURONS)
        opt_state = optax.adam(GeoConfig.LR).init(params)
        h_state = jnp.zeros((GeoConfig.BATCH_SIZE, GeoConfig.NUM_NEURONS * 4))

        dummy = jnp.zeros((1, GeoConfig.SEQ_LEN), dtype=jnp.int32)
        train_step(params, opt_state, h_state, dummy, dummy)
        queue.put({'type': 'READY'})

        epoch = 0; ptr = 0; t0 = time.time()
        
        while not stop_event.is_set():
            if ptr + GeoConfig.SEQ_LEN + 1 > len(raw_data): ptr = 0; epoch += 1
            
            x_batch = jax.lax.dynamic_slice(data_gpu, (ptr,), (GeoConfig.SEQ_LEN,)).reshape(1, GeoConfig.SEQ_LEN)
            y_batch = jax.lax.dynamic_slice(data_gpu, (ptr+1,), (GeoConfig.SEQ_LEN,)).reshape(1, GeoConfig.SEQ_LEN)
            
            params, opt_state, h_state, loss = train_step(params, opt_state, h_state, x_batch, y_batch)
            ptr += GeoConfig.SEQ_LEN
            
            if ptr % (GeoConfig.SEQ_LEN * 2) == 0:
                # Extract Diagonal Activity
                r = np.array(jnp.diag(params['q_rec']['r']))
                i = np.array(jnp.diag(params['q_rec']['i']))
                j = np.array(jnp.diag(params['q_rec']['j']))
                k = np.array(jnp.diag(params['q_rec']['k']))
                packet = {'loss': float(loss), 'q': {'r': r, 'i': i, 'j': j, 'k': k}, 'points': points_np, 'wires': connections}
                try: queue.put_nowait({'type': 'DATA', 'data': packet})
                except mp.queues.Full: pass
    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 2. VGA CANVAS ENGINE
# ==============================================================================
class VGABrain:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1))
        self.layout["M"].split_row(Layout(name="L", ratio=1), Layout(name="R", ratio=1))
        
        self.angle = 0.0
        self.last_data = None

    def get_panel_dims(self):
        """
        Calculates the exact pixel dimensions available in the split panels.
        """
        term_w, term_h = shutil.get_terminal_size()
        
        # Available width for one panel (half screen - padding)
        w = max(10, (term_w // 2) - 4)
        
        # Available height (full screen height - header - padding) * 2 for subpixel density
        h = max(10, (term_h - 4) * 2) 
        
        return w, h

    def render_sphere(self, d, w, h):
        img = Image.new('RGB', (w, h), (5, 5, 15)) 
        draw = ImageDraw.Draw(img)
        
        cx, cy = w / 2, h / 2
        min_dim = min(w, h)
        
        # SCALING FACTORS
        SCALE = min_dim * 0.40          # 80% of smallest dimension
        NODE_SZ = max(1, min_dim * 0.008) # Dynamic node radius (small on small screens, big on big)
        
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        
        points = d['points']
        r_vals = d['q']['r']
        v_min, v_max = r_vals.min(), r_vals.max()
        
        proj = []
        for idx, (x, y, z) in enumerate(points):
            rx = x * cos_a - z * sin_a
            ry = y
            rz = x * sin_a + z * cos_a
            
            px = cx + (rx * SCALE)
            py = cy + (ry * SCALE)
            
            v = (r_vals[idx] - v_min) / (v_max - v_min + 1e-6)
            proj.append({'x': px, 'y': py, 'z': rz, 'v': v, 'idx': idx})
            
        # Wires
        pmap = {p['idx']: p for p in proj}
        for i1, i2 in d['wires']:
            p1 = pmap[i1]
            p2 = pmap[i2]
            
            if p1['z'] > -0.5 and p2['z'] > -0.5: 
                depth = (p1['z'] + p2['z']) / 2
                b = int((depth + 1.0) * 100) + 50
                b = max(50, min(255, b))
                # Simple line, standard width (looks better at terminal density)
                draw.line([(p1['x'], p1['y']), (p2['x'], p2['y'])], fill=(0, b, b))
                
        # Nodes (Z-sorted)
        proj.sort(key=lambda k: k['z'])
        for p in proj:
            if p['z'] > 0: 
                # Depth scaling for 3D effect
                z_mult = 1.0 if p['z'] > 0.5 else 0.7
                sz = max(1, int(NODE_SZ * z_mult))
                
                val = int(p['v'] * 255)
                col = (255, 255-val, 255)
                draw.ellipse([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=col)
                
        return img

    def render_lattice(self, d, w, h):
        img = Image.new('RGB', (w, h), (10, 10, 10)) 
        draw = ImageDraw.Draw(img)
        
        COLS = 32
        ROWS = 32
        
        r = d['q']['r']
        i_v = d['q']['i']
        j_v = d['q']['j']
        k_v = d['q']['k']
        
        v_min, v_max = r.min(), r.max()
        norm_r = (r - v_min) / (v_max - v_min + 1e-6)
        
        def cnorm(val): return int(min(255, max(0, (val * 10 + 0.5) * 255)))
        
        # Calculate padding based on cell size to prevent big gaps
        cell_pixel_w = w / COLS
        PAD = 1 if cell_pixel_w > 10 else 0 
        
        for idx in range(1024):
            c = idx % COLS
            row = idx // COLS
            
            # Use integer math for tight packing
            x0 = int(c * w / COLS)
            y0 = int(row * h / ROWS)
            x1 = int((c + 1) * w / COLS)
            y1 = int((row + 1) * h / ROWS)
            
            bright = 0.4 + norm_r[idx] * 0.6
            red = int(cnorm(i_v[idx]) * bright)
            grn = int(cnorm(j_v[idx]) * bright)
            blu = int(cnorm(k_v[idx]) * bright)
            
            # Fill with minimal padding
            draw.rectangle([x0+PAD, y0+PAD, x1-PAD, y1-PAD], fill=(red, grn, blu))
            
        return img

    def update(self, d):
        if d: self.last_data = d
        if not self.last_data: return self.layout
        
        self.angle += 0.04
        
        head = f"[bold green]WUBU VGA CORE[/bold green] | LOSS: {self.last_data['loss']:.4f}"
        self.layout["T"].update(Panel(head, style="white on black"))
        
        w, h = self.get_panel_dims()
        
        img_3d = self.render_sphere(self.last_data, w, h)
        img_2d = self.render_lattice(self.last_data, w, h)
        
        self.layout["L"].update(Panel(Pixels.from_image(img_3d), title="SPHERE (Dynamic)", style="white on black"))
        self.layout["R"].update(Panel(Pixels.from_image(img_2d), title="LATTICE (1024 Units)", style="white on black"))
        
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=1); stop = mp.Event()
    console = Console(); console.clear()
    
    p = mp.Process(target=island_process, args=(q, stop))
    p.start()
    
    gui = VGABrain(); ready = False
    
    with console.status("[bold green]Booting VGA Interface..."):
        while not ready:
            try:
                m = q.get()
                if m.get('type') == 'READY': ready = True
            except: pass
            
    try:
        with Live(gui.layout, refresh_per_second=30, screen=True) as live:
            while True:
                try:
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA': live.update(gui.update(msg['data']))
                except: 
                    live.update(gui.update(None))
                    time.sleep(0.03)
    except KeyboardInterrupt: stop.set()
    p.join()