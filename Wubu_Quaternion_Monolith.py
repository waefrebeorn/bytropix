# Wubu_Quaternion_Monolith_Fixed.py
# THE HYPERCOMPLEX BRAIN
# STATUS: FIXED (Orthogonal Init Syntax Corrected)
#
# MATH: Quaternion Hamilton Product (Faceted Orthagonics)
# ARCHITECTURE: 4-Channel Holographic RNN
# HARDWARE: RTX 2080 Super

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
# 1. THE QUATERNION ENGINE
# ==============================================================================
class QConfig:
    # 1089 * 1024 = ~1.1M tokens (The Monolith)
    SEQ_LEN = 4096          
    BATCH_SIZE = 1       
    
    # POWER OF 4 CONSTRAINTS
    # Hidden=256 means we actually have 64 Quaternions (64 * 4 = 256)
    HIDDEN_DIM = 2048        
    LR = 0.02               

def island_process(queue, stop_event):
    os.environ['JAX_PLATFORMS'] = '' 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95' 
    
    try:
        import jax
        import jax.numpy as jnp
        import optax
        
        jax.config.update("jax_enable_x64", False) 
        print(f"Island: Allocating Quaternion Lattice on {jax.devices()[0]}")

        # --- QUATERNION MATH KERNEL ---
        def hamilton_product(r, i, j, k, w_r, w_i, w_j, w_k):
            # The Hamilton Rule:
            # r' = rr - ii - jj - kk
            # i' = ri + ir + jk - kj
            # j' = rj - ik + jr + ki
            # k' = rk + ij - ji + kr
            
            o_r = jnp.dot(r, w_r) - jnp.dot(i, w_i) - jnp.dot(j, w_j) - jnp.dot(k, w_k)
            o_i = jnp.dot(r, w_i) + jnp.dot(i, w_r) + jnp.dot(j, w_k) - jnp.dot(k, w_j)
            o_j = jnp.dot(r, w_j) - jnp.dot(i, w_k) + jnp.dot(j, w_r) + jnp.dot(k, w_i)
            o_k = jnp.dot(r, w_k) + jnp.dot(i, w_j) - jnp.dot(j, w_i) + jnp.dot(k, w_r)
            
            return o_r, o_i, o_j, o_k

        def split_quat(tensor):
            chunks = jnp.split(tensor, 4, axis=-1)
            return chunks[0], chunks[1], chunks[2], chunks[3]

        def merge_quat(r, i, j, k):
            return jnp.concatenate([r, i, j, k], axis=-1)

        # --- MODEL INITIALIZATION ---
        def init_q_params(key, dim_in, dim_out):
            q_in = dim_in // 4
            # For orthogonal matrices, dim_in must equal dim_out
            
            keys = jax.random.split(key, 4)
            
            # FIXED: Orthogonal takes an INTEGER size, not a tuple shape
            w_r = jax.random.orthogonal(keys[0], q_in) 
            
            # Imaginary parts initialized small
            w_i = jax.random.normal(keys[1], (q_in, q_in)) * 0.01
            w_j = jax.random.normal(keys[2], (q_in, q_in)) * 0.01
            w_k = jax.random.normal(keys[3], (q_in, q_in)) * 0.01
            
            return {'r': w_r, 'i': w_i, 'j': w_j, 'k': w_k}

        def init_system(key, vocab, hidden):
            k1, k2, k3 = jax.random.split(key, 3)
            return {
                'w_emb': jax.random.normal(k1, (vocab, hidden)) * 0.02,
                # 64x64 Quaternions
                'q_rec': init_q_params(k2, hidden, hidden),
                'w_out': jax.random.normal(k3, (hidden, vocab)) * 0.02,
                'b_rec': jnp.zeros((hidden,)), 
                'b_out': jnp.zeros((vocab,))
            }

        # --- FORWARD PASS ---
        def forward(params, x_seq, h_init):
            
            def step(h, x_idx):
                # 1. Embed
                x = params['w_emb'][x_idx]
                
                # 2. Split
                h_r, h_i, h_j, h_k = split_quat(h)
                x_r, x_i, x_j, x_k = split_quat(x)
                
                # 3. HAMILTON PRODUCT
                wr, wi, wj, wk = params['q_rec']['r'], params['q_rec']['i'], params['q_rec']['j'], params['q_rec']['k']
                rot_r, rot_i, rot_j, rot_k = hamilton_product(h_r, h_i, h_j, h_k, wr, wi, wj, wk)
                
                # 4. Add Input & Bias
                b_r, b_i, b_j, b_k = split_quat(params['b_rec'])
                
                next_r = rot_r + x_r + b_r
                next_i = rot_i + x_i + b_i
                next_j = rot_j + x_j + b_j
                next_k = rot_k + x_k + b_k
                
                # 5. Component-wise Activation
                next_r = jnp.tanh(next_r)
                next_i = jnp.tanh(next_i)
                next_j = jnp.tanh(next_j)
                next_k = jnp.tanh(next_k)
                
                h_new = merge_quat(next_r, next_i, next_j, next_k)
                
                # 6. Project
                logits = jnp.dot(h_new, params['w_out']) + params['b_out']
                return h_new, logits

            def run_row(h_prev, row_x):
                return jax.lax.scan(step, h_prev, row_x)
            
            h_final, logits_seq = jax.vmap(run_row)(h_init, x_seq)
            return h_final, logits_seq

        @jax.jit
        def train_step(params, opt_state, h_state, x_full, y_full):
            def loss_fn(p):
                _, logits = forward(p, x_full, h_state)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                y_flat = y_full.reshape(-1)
                one_hot = jax.nn.one_hot(y_flat, 65)
                log_probs = jax.nn.log_softmax(logits_flat)
                return -jnp.sum(one_hot * log_probs) / y_flat.size

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = optax.adam(QConfig.LR).update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt, jnp.zeros_like(h_state), loss

        @jax.jit
        def generate(params, seed, length=300):
            h = jnp.zeros((QConfig.HIDDEN_DIM,))
            
            def step(carry, _):
                h, idx = carry
                x = params['w_emb'][idx]
                h_r, h_i, h_j, h_k = split_quat(h)
                x_r, x_i, x_j, x_k = split_quat(x)
                wr, wi, wj, wk = params['q_rec']['r'], params['q_rec']['i'], params['q_rec']['j'], params['q_rec']['k']
                rr, ri, rj, rk = hamilton_product(h_r, h_i, h_j, h_k, wr, wi, wj, wk)
                br, bi, bj, bk = split_quat(params['b_rec'])
                
                nr = jnp.tanh(rr + x_r + br)
                ni = jnp.tanh(ri + x_i + bi)
                nj = jnp.tanh(rj + x_j + bj)
                nk = jnp.tanh(rk + x_k + bk)
                
                h_new = merge_quat(nr, ni, nj, nk)
                logits = jnp.dot(h_new, params['w_out']) + params['b_out']
                nxt = jnp.argmax(logits)
                return (h_new, nxt), nxt
                
            def warm(h, x):
                return step((h, x), None)[0][0]
            
            h = jax.lax.scan(lambda c, x: (warm(c, x), None), h, seed)[0]
            
            _, idxs = jax.lax.scan(step, (h, seed[-1]), None, length=length)
            return idxs

        # --- DATA & RUN ---
        print("Island: Loading Monolith Data...")
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Fail " * 200000

        chars = sorted(list(set(text)))
        char_to_ix = {ch:i for i,ch in enumerate(chars)}
        ix_to_char = {i:ch for i,ch in enumerate(chars)}
        raw_data = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)
        
        B, S = QConfig.BATCH_SIZE, QConfig.SEQ_LEN
        fit = B * S
        if len(raw_data)-1 < fit: raw_data = np.pad(raw_data, (0, fit - len(raw_data) + 2))
        
        x_gpu = jax.device_put(jnp.array(raw_data[:fit].reshape(B, S)))
        y_gpu = jax.device_put(jnp.array(raw_data[1:fit+1].reshape(B, S)))

        print("Island: Initializing Quaternion State...")
        key = jax.random.PRNGKey(42)
        params = init_system(key, len(chars), QConfig.HIDDEN_DIM)
        opt_state = optax.adam(QConfig.LR).init(params)
        h_state = jnp.zeros((B, QConfig.HIDDEN_DIM))

        print("Island: Compiling Hamilton Product...")
        # JIT Warmup
        params, opt_state, h_state, loss = train_step(params, opt_state, h_state, x_gpu, y_gpu)
        queue.put({'type': 'READY'})

        epoch = 0
        t0 = time.time()
        while not stop_event.is_set():
            epoch += 1
            params, opt_state, h_state, loss = train_step(params, opt_state, h_state, x_gpu, y_gpu)
            
            elapsed = time.time() - t0
            
            # Generate Text
            seed = x_gpu[0, :32]
            gen_ids = generate(params, seed)
            txt = "".join([ix_to_char[int(i)] for i in gen_ids])
            
            # Pack Quaternion Weights for Vis
            q_weights = {
                'r': np.array(params['q_rec']['r']),
                'i': np.array(params['q_rec']['i']),
                'j': np.array(params['q_rec']['j']),
                'k': np.array(params['q_rec']['k'])
            }
            
            packet = {
                'epoch': epoch,
                'loss': float(loss),
                'time': elapsed,
                'text': txt.replace('\n', ' '),
                'weights': q_weights
            }
            try: queue.put_nowait({'type': 'DATA', 'data': packet})
            except mp.queues.Full: pass
            
    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 2. QUATERNION HOLOGRAPHY (GUI)
# ==============================================================================
class QuatGUI:
    def __init__(self):
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self):
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1))
        self.layout["M"].split_row(Layout(name="Holo", ratio=2), Layout(name="Txt", ratio=1))

    def render_quat_hologram(self, qw):
        r, i, j, k = qw['r'], qw['i'], qw['j'], qw['k']
        
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-6)

        c_r = (norm(i) * 255).astype(np.uint8)
        c_g = (norm(j) * 255).astype(np.uint8)
        c_b = (norm(k) * 255).astype(np.uint8)
        
        intensity = norm(r) 
        
        c_r = (c_r * intensity).astype(np.uint8)
        c_g = (c_g * intensity).astype(np.uint8)
        c_b = (c_b * intensity).astype(np.uint8)
        
        pixels = np.stack([c_r, c_g, c_b], axis=-1)
        
        h, w, _ = pixels.shape
        img = Image.fromarray(pixels)
        img = img.resize((512, 512), resample=Image.NEAREST)
        return Pixels.from_image(img)

    def update(self, d):
        head = f"[bold cyan]WUBU QUATERNION MONOLITH[/bold cyan] | EPOCH: {d['epoch']} | LOSS: {d['loss']:.4f} | RENDER TIME: {d['time']:.2f}s"
        self.layout["T"].update(Panel(head, style="white on black"))
        
        self.layout["Holo"].update(Panel(
            self.render_quat_hologram(d['weights']), 
            title="4D QUATERNION MANIFOLD (RGB=ijk, Brightness=r)", 
            style="white on black"
        ))
        
        self.layout["Txt"].update(Panel(f"[yellow]{d['text']}[/yellow]", title="HYPER-COMPLEX OUTPUT"))
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=1); stop = mp.Event()
    console = Console(); console.clear()
    
    p = mp.Process(target=island_process, args=(q, stop))
    p.start()
    
    gui = QuatGUI(); ready = False
    
    with console.status("[bold green]Allocating Hypercomplex Brain..."):
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