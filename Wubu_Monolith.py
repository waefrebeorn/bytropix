# Wubu_Monolith_Fixed.py
# THE GEODESIC MONOLITH: TOTALITY PROCESSING
# STATUS: FIXED (Orthogonal Init Corrected)
#
# GOAL: Solve Shakespeare in Single-Batch Mode (Global Gradient)
# HARDWARE: RTX 2080 Super (8GB VRAM Capable)
# MATH: Toroidal Geodesic Descent (Exact Manifold)

import multiprocessing as mp
import time
import requests
import numpy as np
import os
import traceback
from typing import NamedTuple

# GUI
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich_pixels import Pixels
from PIL import Image

# ==============================================================================
# 1. THE MONOLITH ENGINE
# ==============================================================================
class MonolithConfig:
    # 1089 rows * 1024 cols = 1,115,136 tokens (The Whole Book)
    SEQ_LEN = 1024          
    BATCH_SIZE = 1089       
    
    HIDDEN_DIM = 256        
    GEAR_RATIO = 20.0       
    FRICTION = 0.999        
    LR = 0.02               

def island_process(queue, stop_event):
    os.environ['JAX_PLATFORMS'] = '' 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95' 
    
    try:
        import jax
        import jax.numpy as jnp
        import optax
        
        # Use 32-bit (float32) for balance of speed/precision. 
        # bfloat16 is risky for "Global Gradient" accumulation.
        jax.config.update("jax_enable_x64", False) 
        
        print(f"Island: Allocating Monolith on {jax.devices()[0]}")

        # --- GEODESIC MATH ---
        class GeodesicState(NamedTuple):
            count: int
            moment1: optax.Updates; moment2: optax.Updates
            stored_topology: optax.Updates; stored_residue: optax.Updates 

        @jax.jit
        def geodesic_update(updates, state, lr):
            gear = MonolithConfig.GEAR_RATIO
            fric = MonolithConfig.FRICTION
            boundary = 2 * jnp.pi 
            
            amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
            quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int32), amplified)
            remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
            
            m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
            m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
            
            cnt = state.count + 1
            m1_hat = optax.bias_correction(m1, 0.9, cnt)
            m2_hat = optax.bias_correction(m2, 0.999, cnt)
            
            final_step = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
            
            new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int32) + q, state.stored_topology, quotients)
            new_res = jax.tree_util.tree_map(lambda s, r: (s * fric) + r, state.stored_residue, remainders)
            
            return final_step, GeodesicState(cnt, m1, m2, new_topo, new_res)

        # --- THE MONOLITH MODEL ---
        def init_params(key, vocab, hidden):
            k1, k2, k3 = jax.random.split(key, 3)
            return {
                'w_emb': jax.random.normal(k1, (vocab, hidden)) * 0.02,
                # FIXED LINE: Orthogonal expects integer dimension for square matrices
                'w_rec': jax.random.orthogonal(k2, hidden), 
                'w_out': jax.random.normal(k3, (hidden, vocab)) * 0.02,
                'b_rec': jnp.zeros((hidden,)),
                'b_out': jnp.zeros((vocab,))
            }

        def forward(params, x_seq, h_init):
            def step(h, x_idx):
                x_emb = params['w_emb'][x_idx]
                h_new = jnp.tanh(jnp.dot(h, params['w_rec']) + x_emb + params['b_rec'])
                logits = jnp.dot(h_new, params['w_out']) + params['b_out']
                return h_new, logits
            
            def run_row(h_prev, row_x):
                return jax.lax.scan(step, h_prev, row_x)
            
            # Vmap processes the 1089 rows in parallel
            h_final, logits_seq = jax.vmap(run_row)(h_init, x_seq)
            return h_final, logits_seq

        @jax.jit
        def monolith_step(params, opt_state, h_state, x_full, y_full):
            # x_full is (1089, 1024)
            
            def loss_fn(p):
                _, logits = forward(p, x_full, h_state)
                # Flattening (Batch * Seq) -> 1.1M tokens
                logits_flat = logits.reshape(-1, logits.shape[-1])
                y_flat = y_full.reshape(-1)
                
                one_hot = jax.nn.one_hot(y_flat, 65)
                log_probs = jax.nn.log_softmax(logits_flat)
                return -jnp.sum(one_hot * log_probs) / y_flat.size

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = geodesic_update(grads, opt_state, MonolithConfig.LR)
            new_params = optax.apply_updates(params, updates)
            
            # Reset H (Stateless Monolith)
            new_h = jnp.zeros_like(h_state)
            
            return new_params, new_opt, new_h, loss

        @jax.jit
        def generate(params, seed, length=300):
            h = jnp.zeros((MonolithConfig.HIDDEN_DIM,))
            def warm(carry, x):
                h = carry
                x_emb = params['w_emb'][x]
                h = jnp.tanh(jnp.dot(h, params['w_rec']) + x_emb + params['b_rec'])
                return h, None
            
            h, _ = jax.lax.scan(warm, h, seed)
            
            def step(carry, _):
                h, idx = carry
                x_emb = params['w_emb'][idx]
                h = jnp.tanh(jnp.dot(h, params['w_rec']) + x_emb + params['b_rec'])
                logits = jnp.dot(h, params['w_out']) + params['b_out']
                nxt = jnp.argmax(logits)
                return (h, nxt), nxt
                
            _, idxs = jax.lax.scan(step, (h, seed[-1]), None, length=length)
            return idxs

        # --- DATA ---
        print("Island: Loading Shakespeare into The Monolith...")
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Fail " * 200000

        chars = sorted(list(set(text)))
        char_to_ix = {ch:i for i,ch in enumerate(chars)}
        ix_to_char = {i:ch for i,ch in enumerate(chars)}
        
        raw_data = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)
        
        # Reshape to Rectangle
        B = MonolithConfig.BATCH_SIZE
        S = MonolithConfig.SEQ_LEN
        fit_len = B * S
        
        if len(raw_data) - 1 < fit_len:
            raw_data = np.pad(raw_data, (0, fit_len - len(raw_data) + 2))
            
        input_data = raw_data[:fit_len].reshape(B, S)
        target_data = raw_data[1:fit_len+1].reshape(B, S)
        
        x_gpu = jax.device_put(jnp.array(input_data))
        y_gpu = jax.device_put(jnp.array(target_data))
        
        print(f"Island: Monolith Constructed. Dims: {x_gpu.shape}. Total Tokens: {x_gpu.size}")

        # --- INIT ---
        key = jax.random.PRNGKey(42)
        params = init_params(key, len(chars), MonolithConfig.HIDDEN_DIM)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        opt_state = GeodesicState(jnp.array(0), zeros, zeros, zeros, zeros)
        h_state = jnp.zeros((B, MonolithConfig.HIDDEN_DIM))

        # --- LOOP ---
        epoch = 0
        t0 = time.time()
        
        print("Island: Compiling Global Gradient Function...")
        params, opt_state, h_state, loss = monolith_step(params, opt_state, h_state, x_gpu, y_gpu)
        queue.put({'type': 'READY'})

        while not stop_event.is_set():
            epoch += 1
            params, opt_state, h_state, loss = monolith_step(params, opt_state, h_state, x_gpu, y_gpu)
            
            elapsed = time.time() - t0
            
            # Generate
            seed = x_gpu[0, :32] 
            gen_ids = generate(params, seed)
            txt = "".join([ix_to_char[int(i)] for i in gen_ids])
            
            # Visualization Data (CPU)
            w_raw = np.array(params['w_rec'])
            
            packet = {
                'epoch': epoch,
                'loss': float(loss),
                'time': elapsed,
                'text': txt.replace('\n', ' '),
                'weights': w_raw
            }
            try: queue.put_nowait({'type': 'DATA', 'data': packet})
            except mp.queues.Full: pass
            
    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 2. THE HOLOGRAM (GUI)
# ==============================================================================
class MonolithGUI:
    def __init__(self):
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self):
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1))
        self.layout["M"].split_row(Layout(name="Holo", ratio=2), Layout(name="Txt", ratio=1))

    def render_hologram(self, weights):
        w = weights
        # Trigonometric Color Mapping
        sig = np.tanh(w) * np.pi 
        r = (np.sin(sig) * 127 + 128).astype(np.uint8)
        g = (np.cos(sig) * 127 + 128).astype(np.uint8)
        b = (np.sin(sig + np.pi/2) * 127 + 128).astype(np.uint8)
        
        pixels = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(pixels)
        img = img.resize((512, 512), resample=Image.NEAREST)
        return Pixels.from_image(img)

    def update(self, d):
        head = f"[bold magenta]WUBU MONOLITH[/bold magenta] | EPOCH: {d['epoch']} (1.1M Tok/step) | LOSS: {d['loss']:.4f} | RUNTIME: {d['time']:.1f}s"
        self.layout["T"].update(Panel(head, style="white on black"))
        
        self.layout["Holo"].update(Panel(self.render_hologram(d['weights']), title="THEORY OF LANGUAGE (WEIGHT MANIFOLD)", style="white on black"))
        self.layout["Txt"].update(Panel(f"[yellow]{d['text']}[/yellow]", title="CRYSTALLIZED OUTPUT"))
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=1); stop = mp.Event()
    console = Console(); console.clear()
    
    p = mp.Process(target=island_process, args=(q, stop))
    p.start()
    
    gui = MonolithGUI(); ready = False
    
    with console.status("[bold cyan]Allocating The Monolith (5GB VRAM)..."):
        while not ready:
            try:
                m = q.get()
                if m.get('type') == 'READY': ready = True
                elif m.get('type') == 'ERROR': 
                    console.print(f"[red]{m['msg']}[/red]"); stop.set(); p.join(); exit()
            except KeyboardInterrupt: stop.set(); p.join(); exit()
            
    try:
        with Live(gui.layout, refresh_per_second=5, screen=True) as live:
            while True:
                try:
                    msg = q.get(timeout=1)
                    if msg['type'] == 'DATA':
                        live.update(gui.update(msg['data']))
                except: pass
    except KeyboardInterrupt: stop.set()
    p.join()