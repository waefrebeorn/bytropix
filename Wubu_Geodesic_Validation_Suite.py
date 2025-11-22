# Wubu_Geodesic_Validation_Suite.py
#
# THE GEODESIC TEST BENCH
# Validates 3 Core Use Cases: Market Regimes, Security Anomalies, Signal Cleaning.
#
# ARCHITECTURE: Async Island (GPU) / Boat (Rich GUI)

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
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich_pixels import Pixels
from PIL import Image

# ==============================================================================
# 1. SCENARIO DATA GENERATORS (Numpy)
# ==============================================================================
def gen_market_spiral(N=50000):
    """Scenario 1: Two entangled spirals (Bull/Bear Regimes)"""
    theta = np.sqrt(np.random.rand(N)) * 720 * (2*np.pi)/360
    d1x = -np.cos(theta)*theta + np.random.rand(N)*0.2
    d1y = np.sin(theta)*theta + np.random.rand(N)*0.2
    X = np.vstack((
        np.hstack((d1x.reshape(-1,1), d1y.reshape(-1,1))), 
        np.hstack((-d1x.reshape(-1,1), -d1y.reshape(-1,1)))
    ))
    Y = np.vstack((np.zeros((N,1)), np.ones((N,1))))
    return X/14.0, Y, "MARKET_CYCLE"

def gen_intrusion_ring(N=50000):
    """Scenario 2: A dense core (Safe) vs Outer Ring (Attack)"""
    # Safe Core (Gaussian)
    safe_n = N // 2
    safe_x = np.random.randn(safe_n, 2) * 0.3
    
    # Attack Ring (Uniform Angle, Fixed Radius)
    attack_n = N - safe_n
    theta = np.random.rand(attack_n) * 2 * np.pi
    r = 1.2 + np.random.randn(attack_n) * 0.1
    attack_x = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    X = np.vstack([safe_x, attack_x])
    Y = np.vstack([np.zeros((safe_n, 1)), np.ones((attack_n, 1))])
    return X/1.5, Y, "SECURITY_CORE"

def gen_signal_grid(N=50000):
    """Scenario 3: The XOR Checkerboard (Discontinuous Signal)"""
    X = (np.random.rand(N, 2) - 0.5) * 4 # Range -2 to 2
    # Logic: sin(x)*sin(y) > 0
    vals = np.sin(X[:, 0]*3) * np.sin(X[:, 1]*3)
    Y = (vals > 0).astype(np.float64).reshape(-1, 1)
    return X/2.0, Y, "SIGNAL_GRID"

# ==============================================================================
# 2. THE ISLAND (TRAINER PROCESS)
# ==============================================================================
def trainer_process(queue, stop_event, scenario_id):
    os.environ['JAX_PLATFORMS'] = '' # GPU Mode
    import jax
    import jax.numpy as jnp
    import optax
    
    jax.config.update("jax_enable_x64", True)
    
    # --- GEODESIC OPTIMIZER ---
    class GeodesicState(NamedTuple):
        count: int; moment1: optax.Updates; moment2: optax.Updates
        stored_topology: optax.Updates; stored_residue: optax.Updates 

    @jax.jit
    def geodesic_update(updates, state, lr):
        gear, fric, boundary = 100.0, 0.99, 2*jnp.pi
        amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
        new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int64) + q, state.stored_topology, quotients)
        new_res = jax.tree_util.tree_map(lambda s, r: (s * fric) + r, state.stored_residue, remainders)
        m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        cnt = state.count + 1
        m1_hat = optax.bias_correction(m1, 0.9, cnt); m2_hat = optax.bias_correction(m2, 0.999, cnt)
        final = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        return final, GeodesicState(cnt, m1, m2, new_topo, new_res)

    # --- MODEL & FUNCTIONS ---
    def init_layer(key, n_in, n_out):
        return {'w': jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2/n_in), 'b': jnp.zeros((n_out,))}

    def forward(params, x):
        h = jnp.tanh(jnp.dot(x, params[0]['w']) + params[0]['b'])
        h = jnp.tanh(jnp.dot(h, params[1]['w']) + params[1]['b']) # 64
        h = jnp.tanh(jnp.dot(h, params[2]['w']) + params[2]['b']) # 64
        h = jnp.tanh(jnp.dot(h, params[3]['w']) + params[3]['b']) # 64 (Deep)
        return jnp.dot(h, params[4]['w']) + params[4]['b']

    @jax.jit
    def train_step(params, opt_state, x, y, lr):
        def loss_fn(p): return jnp.mean((jax.nn.sigmoid(forward(p, x)) - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = geodesic_update(grads, opt_state, lr)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def compute_acc(params, x, y):
        return jnp.mean((jax.nn.sigmoid(forward(params, x)) > 0.5) == y)

    @jax.jit
    def infer_grid(params, grid): return jax.nn.sigmoid(forward(params, grid))

    # --- SETUP ---
    key = jax.random.PRNGKey(42)
    k = jax.random.split(key, 10)
    # Deep Net: 2 -> 64 -> 64 -> 64 -> 64 -> 1
    params = [init_layer(k[0],2,64), init_layer(k[1],64,64), init_layer(k[2],64,64), init_layer(k[3],64,64), init_layer(k[4],64,1)]
    opt_state = GeodesicState(jnp.array(0), *[jax.tree_util.tree_map(jnp.zeros_like, params)]*4)

    # --- SELECT SCENARIO ---
    if scenario_id == 1: X_np, Y_np, tag = gen_market_spiral()
    elif scenario_id == 2: X_np, Y_np, tag = gen_intrusion_ring()
    else: X_np, Y_np, tag = gen_signal_grid()

    X, Y = jnp.array(X_np), jnp.array(Y_np)
    
    # Grid
    res = 64
    grid = jnp.array(np.c_[np.meshgrid(np.linspace(-1.1,1.1,res), np.linspace(-1.1,1.1,res))].reshape(2,-1).T)
    
    # --- LOOP ---
    step = 0
    t0 = time.time()
    while not stop_event.is_set():
        params, opt_state, loss = train_step(params, opt_state, X, Y, 0.002)
        step += 1
        
        if step % 20 == 0:
            acc = compute_acc(params, X, Y)
            grid_out = infer_grid(params, grid)
            try:
                queue.put_nowait({
                    'step': step, 'loss': float(loss), 'acc': float(acc),
                    'grid': np.array(grid_out).reshape(res, res),
                    'tps': step/(time.time()-t0), 'tag': tag
                })
            except: pass
            
            if acc > 0.9995: 
                queue.put({'step': step, 'loss': float(loss), 'acc': float(acc), 'grid': np.array(grid_out).reshape(res, res), 'tps': 0, 'tag': tag})
                stop_event.set()
                break

# ==============================================================================
# 3. THE BOAT (GUI PROCESS)
# ==============================================================================
class ValidationGUI:
    def __init__(self, scenario_type):
        self.console = Console()
        self.layout = Layout()
        self.scenario = scenario_type
        self.setup_layout()
        
    def setup_layout(self):
        self.layout.split_column(Layout(name="head", size=3), Layout(name="body"))
        self.layout["body"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=2))
        self.layout["left"].split_column(Layout(name="stats", ratio=1), Layout(name="desc", ratio=1))

    def get_desc(self):
        if self.scenario == 1:
            return "[bold cyan]TEST A: MARKET REGIME[/bold cyan]\n\nSimulates entwined Bull/Bear cycles.\nGoal: Separate intertwined spirals.\n[yellow]Use: High-Frequency Trading[/yellow]"
        elif self.scenario == 2:
            return "[bold red]TEST B: INTRUSION DETECT[/bold red]\n\nSimulates Safe Core vs Attack Ring.\nGoal: Wrap manifold around Core.\n[yellow]Use: Network Security[/yellow]"
        else:
            return "[bold green]TEST C: SIGNAL REFINER[/bold green]\n\nSimulates Checkerboard XOR.\nGoal: Clean discontinuous noise.\n[yellow]Use: Data Cleaning[/yellow]"

    def get_pixels(self, grid):
        h, w = grid.shape
        img = Image.new('RGB', (w, h))
        px = img.load()
        for y in range(h):
            for x in range(w):
                v = grid[y, x]
                if self.scenario == 1: # Market: Red/Green
                    px[x, y] = (int((1-v)*255), int(v*255), 0)
                elif self.scenario == 2: # Security: Blue/Red
                    px[x, y] = (int(v*255), 0, int((1-v)*255))
                else: # Signal: Black/White
                    c = int(v*255)
                    px[x, y] = (c, c, c)
        return Pixels.from_image(img.resize((w*2, h), Image.NEAREST))

    def render(self, data):
        self.layout["head"].update(Panel(f"GEODESIC VALIDATION SUITE | MODE: {data['tag']}", style="white on blue"))
        
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan"); table.add_column("Val", style="yellow")
        table.add_row("Step", str(data['step']))
        table.add_row("Loss", f"{data['loss']:.6f}")
        table.add_row("Acc", f"{data['acc']*100:.2f}%")
        table.add_row("TPS", f"{data['tps']:.0f}")
        
        self.layout["stats"].update(Panel(table, title="TELEMETRY"))
        self.layout["desc"].update(Panel(self.get_desc(), title="SCENARIO INFO"))
        self.layout["right"].update(Panel(self.get_pixels(data['grid']), title="MANIFOLD VISUALIZATION", style="black"))
        return self.layout

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    console = Console()
    console.clear()
    console.print(Panel("[bold white]GEODESIC NEURAL NETWORK :: VALIDATION SUITE[/bold white]", style="blue"))
    console.print("Select Validation Scenario:")
    console.print("1. [cyan]Market Regimes[/cyan] (Crypto-Spiral)")
    console.print("2. [red]Intrusion Detection[/red] (Security Core)")
    console.print("3. [green]Signal Refining[/green] (XOR Grid)")
    
    choice = IntPrompt.ask("Enter Choice", choices=["1", "2", "3"], default=1)
    
    q = mp.Queue(maxsize=1)
    stop = mp.Event()
    
    p = mp.Process(target=trainer_process, args=(q, stop, choice))
    p.start()
    
    gui = ValidationGUI(choice)
    
    try:
        with Live(gui.layout, refresh_per_second=30, screen=True) as live:
            while True:
                if stop.is_set() and q.empty(): break
                try:
                    data = q.get_nowait()
                    live.update(gui.render(data))
                except: time.sleep(0.01)
    except KeyboardInterrupt:
        stop.set()
    
    stop.set()
    p.join()
    console.print("[bold green]TEST COMPLETE.[/bold green]")