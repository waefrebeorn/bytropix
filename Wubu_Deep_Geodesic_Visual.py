# Wubu_Island_Boat.py
# THE GEODESIC MLP: ASYNC GPU ARCHITECTURE
#
# PROCESS A (ISLAND): 100% GPU Load, solving the topology.
# PROCESS B (BOAT):   Rich GUI, visualizing the signal.

import multiprocessing as mp
import time
import numpy as np
import os
from typing import NamedTuple

# Visualization Imports (The Boat)
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box
from rich_pixels import Pixels
from PIL import Image

# ==============================================================================
# 1. THE GEODESIC MATH (SHARED LIBRARY)
# ==============================================================================
# We define logic here, but JAX is only imported inside the Island Process
# to prevent CUDA initialization conflicts.

class GeodesicConfig:
    LAYER_SIZES = [2, 64, 64, 64, 1]
    GEAR_RATIO = 100.0  # High precision
    FRICTION = 0.99
    LR = 0.002

# ==============================================================================
# 2. THE ISLAND (TRAINING PROCESS)
# ==============================================================================
def trainer_process(queue, stop_event):
    # --- ISOLATED GPU CONTEXT ---
    os.environ['JAX_PLATFORMS'] = '' # Prefer GPU, fall back to CPU if needed
    import jax
    import jax.numpy as jnp
    import optax
    
    jax.config.update("jax_enable_x64", True)
    
    # --- OPTIMIZER LOGIC (JIT) ---
    class GeodesicState(NamedTuple):
        count: int; moment1: optax.Updates; moment2: optax.Updates
        stored_topology: optax.Updates; stored_residue: optax.Updates 

    @jax.jit
    def geodesic_update(updates, state, lr):
        # The Geodesic Engine
        gear = GeodesicConfig.GEAR_RATIO
        fric = GeodesicConfig.FRICTION
        boundary = 2 * jnp.pi
        
        amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
        
        new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int64) + q, state.stored_topology, quotients)
        new_res = jax.tree_util.tree_map(lambda s, r: (s * fric) + r, state.stored_residue, remainders)
        
        # Adam Core
        m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        
        cnt = state.count + 1
        m1_hat = optax.bias_correction(m1, 0.9, cnt)
        m2_hat = optax.bias_correction(m2, 0.999, cnt)
        final = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        
        return final, GeodesicState(cnt, m1, m2, new_topo, new_res)

    # --- MODEL DEFINITION ---
    def init_layer(key, n_in, n_out):
        w = jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2/n_in)
        b = jnp.zeros((n_out,))
        return {'w': w, 'b': b}

    def forward(params, x):
        # Layer 0
        h = jnp.tanh(jnp.dot(x, params[0]['w']) + params[0]['b'])
        # Layer 1
        h = jnp.tanh(jnp.dot(h, params[1]['w']) + params[1]['b'])
        # Layer 2
        h = jnp.tanh(jnp.dot(h, params[2]['w']) + params[2]['b'])
        # Output
        return jnp.dot(h, params[3]['w']) + params[3]['b']

    @jax.jit
    def train_step(params, opt_state, x, y, lr):
        def loss_fn(p):
            logits = forward(p, x)
            probs = jax.nn.sigmoid(logits)
            return jnp.mean((probs - y) ** 2) # MSE for raw precision
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = geodesic_update(grads, opt_state, lr)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def compute_accuracy(params, x, y):
        logits = forward(params, x)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float64)
        return jnp.mean(preds == y)

    @jax.jit
    def render_grid_inference(params, grid):
        # Fully vectorized grid inference on GPU
        logits = forward(params, grid)
        return jax.nn.sigmoid(logits)

    # --- INITIALIZATION ---
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # [2 -> 64 -> 64 -> 64 -> 1]
    params = [
        init_layer(keys[0], 2, 64),
        init_layer(keys[1], 64, 64),
        init_layer(keys[2], 64, 64),
        init_layer(keys[3], 64, 1)
    ]
    
    # Init Optimizer State
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_state = GeodesicState(jnp.array(0), zeros, zeros, zeros, zeros)

    # --- DATA GENERATION (Two Spirals - 1000 points) ---
    N = 1000000
    theta = np.sqrt(np.random.rand(N)) * 720 * (2*np.pi)/360
    d1x = -np.cos(theta)*theta + np.random.rand(N)*0.1
    d1y = np.sin(theta)*theta + np.random.rand(N)*0.1
    
    X_np = np.vstack((
        np.hstack((d1x.reshape(-1,1), d1y.reshape(-1,1))), 
        np.hstack((-d1x.reshape(-1,1), -d1y.reshape(-1,1)))
    ))
    Y_np = np.vstack((np.zeros((N,1)), np.ones((N,1))))
    
    # Normalize
    X_np = X_np / 14.0
    X = jnp.array(X_np)
    Y = jnp.array(Y_np)

    # Pre-calc Visualization Grid (64x64)
    res_x, res_y = 64, 64
    xs = np.linspace(-1.1, 1.1, res_x)
    ys = np.linspace(-1.1, 1.1, res_y)
    xx, yy = np.meshgrid(xs, ys)
    grid_inputs = jnp.array(np.c_[xx.ravel(), yy.ravel()])

    # --- MAIN LOOP (The Island) ---
    # This loop runs as fast as the GPU allows.
    
    step = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # 1. Train Step (Full Batch for simplicity and stability)
        params, opt_state, loss = train_step(params, opt_state, X, Y, GeodesicConfig.LR)
        step += 1

        # 2. Check Flag Condition (Every 50 steps)
        if step % 50 == 0:
            # Non-blocking check: only do work if someone is listening? 
            # Actually, we just calculate and try to push.
            
            # Run inference on accuracy
            acc = compute_accuracy(params, X, Y)
            
            # Run inference on Grid (GPU)
            grid_out = render_grid_inference(params, grid_inputs)
            
            # Prepare Packet (Move to CPU memory)
            # We do this transfer only once every 50 steps.
            packet = {
                'step': step,
                'loss': float(loss),
                'acc': float(acc),
                'grid': np.array(grid_out).reshape(res_y, res_x),
                'tps': step / (time.time() - start_time)
            }
            
            # THE FLAG: Put data in queue. If full, Drop it!
            # This ensures the Trainer NEVER waits for the GUI.
            try:
                queue.put_nowait(packet)
            except mp.queues.Full:
                pass # Boat is too slow, Island keeps working.

            if acc > 0.9999:
                # One final guaranteed push
                queue.put(packet) 
                stop_event.set()
                break

# ==============================================================================
# 3. THE BOAT (GUI PROCESS)
# ==============================================================================
class BoatGUI:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self):
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1)
        )
        self.layout["body"].split_row(
            Layout(name="telemetry", ratio=1),
            Layout(name="viewport", ratio=2)
        )

    def generate_pixels(self, grid_data):
        # Convert 0.0-1.0 float array to RGB Image
        # 0.0 (Class 0) = Blue/Purple, 1.0 (Class 1) = Red/Orange
        h, w = grid_data.shape
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        
        for y in range(h):
            for x in range(w):
                val = grid_data[y, x]
                # Cyberpunk Heatmap
                r = int(val * 255)
                b = int((1.0 - val) * 255)
                g = int(0) # (val * 50)
                pixels[x, y] = (r, g, b)
        
        # Upscale for Rich Display (Nearest Neighbor to keep blocky look)
        img = img.resize((w*2, h), resample=Image.NEAREST)
        return Pixels.from_image(img)

    def update_view(self, data):
        # Header
        self.layout["header"].update(Panel(
            f"ISLAND LINK: [bold green]CONNECTED[/bold green] | TPS: {data['tps']:.1f} | GEODESIC BATCH MODE", 
            style="white on black"
        ))
        
        # Telemetry
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("SENSOR", style="cyan")
        table.add_column("READING", style="yellow")
        table.add_row("EPOCH", str(data['step']))
        table.add_row("LOSS", f"{data['loss']:.8f}")
        table.add_row("ACCURACY", f"{data['acc']*100:.4f}%")
        table.add_row("TARGET", "99.9900%")
        
        color = "red"
        if data['acc'] > 0.9: color = "yellow"
        if data['acc'] > 0.99: color = "green"
        
        self.layout["telemetry"].update(Panel(
            table, title="[bold]STATS[/bold]", border_style=color
        ))
        
        # Viewport (Pixels)
        px = self.generate_pixels(data['grid'])
        self.layout["viewport"].update(Panel(px, title="TOPOLOGY SCAN", style="white on black"))
        
        return self.layout

# ==============================================================================
# 4. ORCHESTRATION
# ==============================================================================
if __name__ == "__main__":
    # Must use spawn for JAX compatibility
    mp.set_start_method('spawn')
    
    # The Flag (Queue size 1 = Drop old frames immediately)
    communication_queue = mp.Queue(maxsize=1)
    stop_signal = mp.Event()
    
    print(">>> BOOTING ISLAND (GPU)...")
    trainer = mp.Process(target=trainer_process, args=(communication_queue, stop_signal))
    trainer.start()
    
    print(">>> LAUNCHING BOAT (GUI)...")
    gui = BoatGUI()
    
    last_data = None
    
    try:
        with Live(gui.layout, refresh_per_second=60, screen=True) as live:
            while True:
                # Check if training is done
                if stop_signal.is_set() and communication_queue.empty():
                    break
                
                # Try to get new flag from Island
                try:
                    # Non-blocking get. 
                    # If empty, we just loop and re-render last frame (or wait)
                    data = communication_queue.get_nowait()
                    last_data = data
                    live.update(gui.update_view(data))
                except mp.queues.Empty:
                    # No new flag. Island is busy mathing. 
                    # Don't update, just sleep a tiny bit to save CPU on the Boat.
                    time.sleep(0.01)
                    
    except KeyboardInterrupt:
        print(">>> MANUAL OVERRIDE.")
        stop_signal.set()
    
    # Clean up
    print(">>> DOCKING...")
    stop_signal.set()
    trainer.join()
    
    if last_data:
        print(f"\nFINAL STATS :: Steps: {last_data['step']} | Acc: {last_data['acc']*100:.4f}%")
        if last_data['acc'] > 0.999:
            print("MISSION SUCCESS: 99.99% CONVERGENCE REACHED.")
        else:
            print("MISSION ABORTED.")