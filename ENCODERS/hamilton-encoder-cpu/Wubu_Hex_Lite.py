# Wubu_Shakespeare_Toroidal_Fixed.py
# THE GEODESIC CORTEX: TOROIDAL NLP
#
# VISUALIZATION FIX: Trigonometric Phase Mapping (Sin/Cos embedding)
# MATH: Toroidal Geodesic Topology (Pi/-Pi Wrapping)

import multiprocessing as mp
import time
import requests
import numpy as np
import os
from typing import NamedTuple

# GUI Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box
from rich_pixels import Pixels
from PIL import Image

# ==============================================================================
# 1. THE GEODESIC MATH (THE HOLY GRAIL)
# ==============================================================================
class GeodesicConfig:
    BATCH_SIZE = 1        
    SEQ_LEN = 256           
    HIDDEN_DIM = 256        
    GEAR_RATIO = 50.0       
    FRICTION = 0.995        
    LR = 0.005              

# ==============================================================================
# 2. THE ISLAND (CORTEX TRAINING)
# ==============================================================================
def cortex_process(queue, stop_event):
    os.environ['JAX_PLATFORMS'] = '' 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    import jax
    import jax.numpy as jnp
    import optax
    
    jax.config.update("jax_enable_x64", True)
    
    print(f"Island: Geodesic Cortex active on {jax.devices()[0]}")

    class GeodesicState(NamedTuple):
        count: int
        moment1: optax.Updates; moment2: optax.Updates
        stored_topology: optax.Updates; stored_residue: optax.Updates 

    @jax.jit
    def geodesic_update(updates, state, lr):
        gear = GeodesicConfig.GEAR_RATIO
        fric = GeodesicConfig.FRICTION
        boundary = 2 * jnp.pi 
        
        amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
        
        # Toroidal Wrap
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
        
        new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int64) + q, state.stored_topology, quotients)
        
        m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        
        cnt = state.count + 1
        m1_hat = optax.bias_correction(m1, 0.9, cnt)
        m2_hat = optax.bias_correction(m2, 0.999, cnt)
        
        final_step = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        new_res = jax.tree_util.tree_map(lambda s, r: (s * fric) + r, state.stored_residue, remainders)
        
        return final_step, GeodesicState(cnt, m1, m2, new_topo, new_res)

    # --- MODEL (TOROIDAL RNN) ---
    def init_params(key, vocab_size, hidden):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # Initialize STRICTLY -Pi to Pi
        scale = jnp.pi / jnp.sqrt(hidden)
        return {
            'w_emb': jax.random.uniform(k1, (vocab_size, hidden), minval=-jnp.pi, maxval=jnp.pi),
            'w_rec': jax.random.uniform(k2, (hidden, hidden), minval=-scale, maxval=scale),
            'w_out': jax.random.uniform(k3, (hidden, vocab_size), minval=-scale, maxval=scale),
            'b_rec': jnp.zeros((hidden,)),
            'b_out': jnp.zeros((vocab_size,))
        }

    def forward(params, x_seq, h_prev):
        def step(h, x_idx):
            x_emb = params['w_emb'][x_idx]
            # Math: Phase Addition
            pre_act = jnp.dot(h, params['w_rec']) + x_emb + params['b_rec']
            # Tanh keeps us in valid manifold phase (-1 to 1)
            h_new = jnp.tanh(pre_act) 
            logits = jnp.dot(h_new, params['w_out']) + params['b_out']
            return h_new, logits

        h_final, logits_seq = jax.lax.scan(step, h_prev, x_seq)
        return h_final, logits_seq

    @jax.jit
    def train_step(params, opt_state, h_prev, x_batch, y_batch):
        def loss_fn(p):
            def run_seq(h, x, y):
                _, logits = forward(p, x, h)
                one_hot = jax.nn.one_hot(y, 65) 
                log_probs = jax.nn.log_softmax(logits)
                return -jnp.sum(one_hot * log_probs)
            losses = jax.vmap(run_seq)(h_prev, x_batch, y_batch)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = geodesic_update(grads, opt_state, GeodesicConfig.LR)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def generate_text(params, seed_idxs, length=200):
        h = jnp.zeros((GeodesicConfig.HIDDEN_DIM,))
        def step(carry, _):
            h, curr_idx = carry
            h, logits = forward(params, jnp.array([curr_idx]), h)
            next_idx = jnp.argmax(logits[0])
            return (h, next_idx), next_idx

        for i in range(len(seed_idxs)):
            h, _ = forward(params, jnp.array([seed_idxs[i]]), h)
        
        curr = seed_idxs[-1]
        _, gen_idxs = jax.lax.scan(step, (h, curr), None, length=length)
        return gen_idxs

    # --- DATA ---
    print("Island: Acquiring Shakespeare...")
    try: url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"; text = requests.get(url).text
    except: text = "To be or not to be " * 1000

    chars = sorted(list(set(text)))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}
    data_indices = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)
    
    key = jax.random.PRNGKey(1337)
    params = init_params(key, len(chars), GeodesicConfig.HIDDEN_DIM)
    
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_state = GeodesicState(jnp.array(0), zeros, zeros, zeros, zeros)
    h_init = jnp.zeros((GeodesicConfig.BATCH_SIZE, GeodesicConfig.HIDDEN_DIM))

    step = 0; ptr = 0; t0 = time.time()
    
    while not stop_event.is_set():
        if ptr + GeodesicConfig.BATCH_SIZE * GeodesicConfig.SEQ_LEN + 1 > len(data_indices): ptr = 0
            
        idxs = []; targets = []
        for i in range(GeodesicConfig.BATCH_SIZE):
            p = ptr + i * GeodesicConfig.SEQ_LEN
            idxs.append(data_indices[p : p + GeodesicConfig.SEQ_LEN])
            targets.append(data_indices[p+1 : p + GeodesicConfig.SEQ_LEN + 1])
        
        x_batch = jnp.array(np.stack(idxs)); y_batch = jnp.array(np.stack(targets))
        ptr += GeodesicConfig.BATCH_SIZE * GeodesicConfig.SEQ_LEN

        params, opt_state, loss = train_step(params, opt_state, h_init, x_batch, y_batch)
        step += 1

        if step % 50 == 0:
            seed = np.array([char_to_ix['T'], char_to_ix['h'], char_to_ix['e']])
            gen_ids = generate_text(params, seed)
            gen_txt = "".join([ix_to_char[int(i)] for i in gen_ids])
            
            # SEND RAW PHASE ANGLES TO BOAT
            w_phase = np.array(params['w_rec'], dtype=np.float32)
            
            packet = {'step': step, 'loss': float(loss), 'tps': step / (time.time()-t0), 'text': gen_txt.replace('\n', ' '), 'phase_map': w_phase}
            try: queue.put_nowait(packet)
            except mp.queues.Full: pass

# ==============================================================================
# 3. THE BOAT (GUI PROCESS)
# ==============================================================================
class TorusGUI:
    def __init__(self):
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self):
        self.layout.split_column(Layout(name="header", size=3), Layout(name="main", ratio=1))
        self.layout["main"].split_row(Layout(name="stats", ratio=1), Layout(name="vis", ratio=3))
        self.layout["vis"].split_column(Layout(name="heatmap", ratio=2), Layout(name="text", ratio=1))

    def generate_heatmap(self, weights):
        # 1. TOROIDAL CLAMPING (Wrap to -Pi..Pi)
        # TGT Logic: The manifold is a circle.
        wrapped = (weights + np.pi) % (2 * np.pi) - np.pi
        
        # 2. AUTO-EXPOSURE (Zoom into small ripples if variance is low)
        # This prevents the "Solid Green" bug.
        std = np.std(wrapped)
        boost = 1.0
        if std < 1.0:
            boost = np.pi / (std * 4 + 1e-6) # Aggressive Zoom
            
        w_zoomed = wrapped * boost
        
        # 3. TRIGONOMETRIC COLOR MAPPING (The "Math Renderer")
        # We define color NOT by linear value, but by angular phase.
        # This creates a perfect cycle where -Pi and Pi meet visually.
        
        # Red   = Sine(Theta)
        # Green = Cosine(Theta)
        # Blue  = Sine(Theta + PhaseShift)
        
        r = (np.sin(w_zoomed) * 127 + 128).astype(np.uint8)
        g = (np.cos(w_zoomed) * 127 + 128).astype(np.uint8)
        b = (np.sin(w_zoomed + 2.0) * 127 + 128).astype(np.uint8)
        
        # Assemble Image
        h, w = weights.shape
        pixels = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(pixels)
        img = img.resize((w*2, h), resample=Image.NEAREST)
        return Pixels.from_image(img)

    def update(self, data):
        self.layout["header"].update(Panel(f"WUBU TOROIDAL CORTEX | STEP: {data['step']} | LOSS: {data['loss']:.4f} | TPS: {data['tps']:.1f}", style="bold white on blue"))
        
        self.layout["stats"].update(Panel(
            f"[bold]TGT MANIFOLD[/bold]\n\n"
            f"Mode: [green]Geodesic TGT[/green]\n"
            f"Friction: 0.995\n"
            f"Phase Wrap: Active\n\n"
            f"[dim]Visuals derived directly\nfrom Phase Angles[/dim]",
            title="ENGINE"
        ))
        
        hm = self.generate_heatmap(data['phase_map'])
        self.layout["heatmap"].update(Panel(hm, title="NEURAL PHASE MANIFOLD (Sin/Cos Mapping)", style="white on black"))
        
        self.layout["text"].update(Panel(f"[yellow]{data['text']}[/yellow]", title="SHAKESPEARE GENERATOR"))
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=1); stop = mp.Event()
    trainer = mp.Process(target=cortex_process, args=(q, stop)); trainer.start()
    gui = TorusGUI()
    
    try:
        with Live(gui.layout, refresh_per_second=10, screen=True) as live:
            while True:
                if stop.is_set() and q.empty(): break
                try:
                    data = q.get(timeout=1.0)
                    live.update(gui.update(data))
                except: pass
    except KeyboardInterrupt: stop.set()
    trainer.join()