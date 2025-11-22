# Wubu_Geodesic_Benchmark_Pro.py
#
# THE GEODESIC BENCHMARK PRO
# 1. Harder Data.
# 2. Correct Aspect Ratio Visuals.
# 3. 100% GPU Saturation.

import multiprocessing as mp
import time
import numpy as np
import os
import sys
from typing import NamedTuple

# GUI Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box
from rich.prompt import IntPrompt
from rich_pixels import Pixels
from PIL import Image

# ==============================================================================
# 1. DATA GENERATORS (Tuned)
# ==============================================================================
def gen_broken_horseshoes(N=50000):
    """TEST 1: MARKET REGIME"""
    n_class = N // 2
    # Class 0 (Top U)
    t0 = np.linspace(0, np.pi, n_class) + np.random.randn(n_class)*0.05
    x0 = np.cos(t0) - 0.5; y0 = np.sin(t0) + 0.2
    # Class 1 (Bottom U)
    t1 = np.linspace(0, np.pi, n_class) + np.random.randn(n_class)*0.05
    x1 = np.cos(t1) + 0.5; y1 = -np.sin(t1) - 0.2
    
    X = np.vstack((np.column_stack((x0, y0)), np.column_stack((x1, y1))))
    Y = np.concatenate((np.zeros(n_class), np.ones(n_class))).reshape(-1, 1)
    # Noise Background
    noise = (np.random.rand(N//10, 2) - 0.5) * 3
    X = np.vstack((X, noise))
    Y = np.vstack((Y, np.zeros((N//10, 1))))
    return X, Y, "MARKET_HORSESHOE"

def gen_multi_cluster(N=50000):
    """TEST 2: INTRUSION (Tighter Clusters)"""
    n_safe = N // 3
    n_attack = N - n_safe
    # 3 Distinct Tight Clusters
    c1 = np.random.randn(n_safe//3, 2) * 0.10 + np.array([-0.6, -0.6])
    c2 = np.random.randn(n_safe//3, 2) * 0.10 + np.array([0.6, 0.0])
    c3 = np.random.randn(n_safe//3, 2) * 0.10 + np.array([-0.6, 0.6])
    X_safe = np.vstack((c1, c2, c3))
    X_attack = (np.random.rand(n_attack, 2) - 0.5) * 4
    X = np.vstack((X_safe, X_attack))
    Y = np.vstack((np.zeros((len(X_safe), 1)), np.ones((len(X_attack), 1))))
    return X/1.2, Y, "SECURITY_CLUSTERS"

def gen_high_freq_xor(N=50000):
    """TEST 3: SIGNAL (Checkerboard)"""
    X = (np.random.rand(N, 2) - 0.5) * 5
    vals = np.sin(X[:, 0]*2.5) * np.sin(X[:, 1]*2.5)
    Y = (vals > 0).astype(np.float64).reshape(-1, 1)
    return X/2.5, Y, "SIGNAL_CHECKERBOARD"

# ==============================================================================
# 2. ISLAND (GPU)
# ==============================================================================
def trainer_process(queue, stop_event, scenario_id):
    os.environ['JAX_PLATFORMS'] = '' 
    import jax
    import jax.numpy as jnp
    import optax
    jax.config.update("jax_enable_x64", True)

    # --- JIT MATH ---
    class GeodesicState(NamedTuple):
        count: int; moment1: optax.Updates; moment2: optax.Updates
        stored_topology: optax.Updates; stored_residue: optax.Updates 

    @jax.jit
    def geodesic_update(updates, state, lr):
        gear, fric = 120.0, 0.99 
        boundary = 2 * jnp.pi
        amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
        new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int64) + q, state.stored_topology, quotients)
        new_res = jax.tree_util.tree_map(lambda s, r: (s * fric) + r, state.stored_residue, remainders)
        m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        cnt = state.count + 1
        final = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), 
                                       optax.bias_correction(m1, 0.9, cnt), optax.bias_correction(m2, 0.999, cnt))
        return final, GeodesicState(cnt, m1, m2, new_topo, new_res)

    def init_layer(key, n_in, n_out):
        return {'w': jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2/n_in), 'b': jnp.zeros((n_out,))}

    def forward(params, x):
        h = x
        for i in range(len(params)-1):
            h = jnp.tanh(jnp.dot(h, params[i]['w']) + params[i]['b'])
        return jnp.dot(h, params[-1]['w']) + params[-1]['b']

    @jax.jit
    def train_step(params, opt_state, x, y, lr):
        def loss_fn(p): return jnp.mean((jax.nn.sigmoid(forward(p, x)) - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = geodesic_update(grads, opt_state, lr)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def infer_grid(params, grid): 
        return jax.nn.sigmoid(forward(params, grid))

    # --- INIT ---
    key = jax.random.PRNGKey(99)
    k = jax.random.split(key, 10)
    # [2 -> 128 -> 128 -> 128 -> 1]
    params = [init_layer(k[0],2,128), init_layer(k[1],128,128), init_layer(k[2],128,128), init_layer(k[3],128,1)]
    opt_state = GeodesicState(jnp.array(0), *[jax.tree_util.tree_map(jnp.zeros_like, params)]*4)

    if scenario_id == 1: X_np, Y_np, tag = gen_broken_horseshoes()
    elif scenario_id == 2: X_np, Y_np, tag = gen_multi_cluster()
    else: X_np, Y_np, tag = gen_high_freq_xor()
    X, Y = jnp.array(X_np), jnp.array(Y_np)

    # --- CORRECT GRID GENERATION ---
    res = 128
    # Explicit Meshgrid
    xs = np.linspace(-1.5, 1.5, res)
    ys = np.linspace(-1.5, 1.5, res)
    xx, yy = np.meshgrid(xs, ys) # xx varies col, yy varies row
    
    # Flatten correctly: (x, y) pairs
    # We must ravel such that reshaping back to (res, res) preserves the image
    grid_flat = np.column_stack([xx.ravel(), yy.ravel()]) 
    grid_inputs = jnp.array(grid_flat)

    step = 0
    t0 = time.time()
    while not stop_event.is_set():
        params, opt_state, loss = train_step(params, opt_state, X, Y, 0.002)
        step += 1
        
        if step % 40 == 0:
            # Calculate Acc on Batch
            logits = forward(params, X)
            acc = jnp.mean((jax.nn.sigmoid(logits) > 0.5) == Y)
            
            # Visual Inference
            grid_out = infer_grid(params, grid_inputs)
            # Reshape back to image dimensions (H, W)
            grid_img = np.array(grid_out).reshape(res, res)
            
            try:
                queue.put_nowait({
                    'step': step, 'loss': float(loss), 'acc': float(acc),
                    'grid': grid_img, 'tps': step/(time.time()-t0), 'tag': tag
                })
            except: pass
            
            if acc > 0.9990: 
                queue.put({'step': step, 'loss': float(loss), 'acc': float(acc), 'grid': grid_img, 'tps': 0, 'tag': tag})
                stop_event.set()
                break

# ==============================================================================
# 3. BOAT (VISUALIZER)
# ==============================================================================
class VisualFixGUI:
    def __init__(self, scenario):
        self.scenario = scenario
        self.layout = Layout()
        self.layout.split_column(Layout(name="top", size=3), Layout(name="main"))
        self.layout["main"].split_row(Layout(name="vis", ratio=2), Layout(name="info", ratio=1))

    def get_pixels(self, grid):
        # Grid is (H, W) float 0..1
        # Vectorized Color Mapping (No loops!)
        h, w = grid.shape
        
        # Create RGB array
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        v = grid
        
        if self.scenario == 1: # Market: Cyan (0) to Magenta (1)
            # Red channel: 0 -> 255
            rgb[..., 0] = (v * 255).astype(np.uint8)
            # Green channel: 255 -> 0
            rgb[..., 1] = ((1 - v) * 255).astype(np.uint8)
            # Blue channel: 255 -> 255
            rgb[..., 2] = 255
            
        elif self.scenario == 2: # Intrusion: Blue (Safe) to Red (Attack)
            # Red: 0 -> 255
            rgb[..., 0] = (v * 255).astype(np.uint8)
            # Green: 0
            rgb[..., 1] = 0
            # Blue: 255 -> 0
            rgb[..., 2] = ((1 - v) * 255).astype(np.uint8)
            
        else: # Signal: Black to Green
            # Green channel only
            rgb[..., 1] = (v * 255).astype(np.uint8)

        img = Image.fromarray(rgb, mode='RGB')
        # Scale for Terminal (Width x 2)
        img = img.resize((w*2, h), Image.NEAREST)
        return Pixels.from_image(img)

    def render(self, data):
        self.layout["top"].update(Panel(f"GEODESIC VISUAL FIX :: {data['tag']}", style="white on blue"))
        
        t = Table(box=box.HEAVY_EDGE)
        t.add_column("METRIC"); t.add_column("VAL")
        t.add_row("STEP", str(data['step']))
        t.add_row("LOSS", f"{data['loss']:.5f}")
        t.add_row("ACC", f"{data['acc']*100:.2f}%")
        
        self.layout["info"].update(Panel(t, title="STATS"))
        self.layout["vis"].update(Panel(self.get_pixels(data['grid']), title="MANIFOLD SCAN", style="black"))
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    console = Console()
    console.clear()
    
    console.print(Panel("[bold green]GEODESIC VISUAL FIX[/bold green]\n1. Broken Horseshoes\n2. Security Clusters\n3. Signal Checkerboard", style="blue"))
    choice = IntPrompt.ask("SELECT", choices=["1", "2", "3"], default=1)
    
    q = mp.Queue(maxsize=1)
    stop = mp.Event()
    
    p = mp.Process(target=trainer_process, args=(q, stop, choice))
    p.start()
    
    gui = VisualFixGUI(choice)
    
    try:
        with Live(gui.layout, refresh_per_second=30, screen=True) as live:
            while True:
                if stop.is_set() and q.empty(): break
                try:
                    d = q.get_nowait()
                    live.update(gui.render(d))
                except: time.sleep(0.01)
    except KeyboardInterrupt: pass
    
    stop.set()
    p.join()