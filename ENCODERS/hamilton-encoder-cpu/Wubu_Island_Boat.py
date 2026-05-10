# Wubu_Island_Boat_Optimized.py
# ARCHITECTURE: ASYNC GPU GEODESIC MLP
# HARDWARE TARGET: NVIDIA RTX 2080 Super
# OPTIMIZATION: Mini-Batch Streamlined

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
# 1. CONFIGURATION & SHARED MATH
# ==============================================================================

class GeodesicConfig:
    # Hardware Optimization
    BATCH_SIZE = 2048       # Set to 1 for pure SGD, but 2048 is 1000x faster on 2080 Super
    N_SAMPLES = 1_000_000   # Total points (500k per class)
    
    # Model Hparams
    LAYER_SIZES = [2, 128, 128, 64, 1] # Slightly wider for better convergence
    GEAR_RATIO = 100.0
    FRICTION = 0.99
    LR = 0.002

# ==============================================================================
# 2. THE ISLAND (TRAINING PROCESS)
# ==============================================================================
def trainer_process(queue, stop_event):
    # --- GPU CONTEXT SETUP ---
    # Prevent JAX from pre-allocating 90% of memory so the Boat (GUI) doesn't lag
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85' 
    
    import jax
    import jax.numpy as jnp
    import optax
    
    # High precision is required for the Geodesic remainder logic
    jax.config.update("jax_enable_x64", True)
    
    print(f"Island: JAX initialized on {jax.devices()[0]}")

    # --- OPTIMIZER (GEODESIC) ---
    class GeodesicState(NamedTuple):
        count: int
        moment1: optax.Updates
        moment2: optax.Updates
        stored_topology: optax.Updates
        stored_residue: optax.Updates 

    @jax.jit
    def geodesic_update(updates, state, lr):
        gear = GeodesicConfig.GEAR_RATIO
        fric = GeodesicConfig.FRICTION
        boundary = 2 * jnp.pi
        
        # 1. Amplify Gradients
        amplified = jax.tree_util.tree_map(lambda g: g * gear, updates)
        
        # 2. Split into Topological Quotient and Residue
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
        
        # 3. Store Topology (Long-term memory)
        new_topo = jax.tree_util.tree_map(lambda s, q: (s * fric).astype(jnp.int64) + q, state.stored_topology, quotients)
        
        # 4. Adam Optimization on the Residue (Short-term precision)
        m1 = optax.incremental_update(jax.tree_util.tree_map(lambda r: r/gear, remainders), state.moment1, 0.9)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        
        cnt = state.count + 1
        m1_hat = optax.bias_correction(m1, 0.9, cnt)
        m2_hat = optax.bias_correction(m2, 0.999, cnt)
        
        final_updates = jax.tree_util.tree_map(lambda m1, m2: -lr * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        
        return final_updates, GeodesicState(cnt, m1, m2, new_topo, state.stored_residue)

    # --- MODEL ---
    def init_layer(key, n_in, n_out):
        # Xavier Initialization
        return {
            'w': jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2/n_in),
            'b': jnp.zeros((n_out,))
        }

    def forward(params, x):
        h = x
        # Unroll layers manually for speed
        h = jnp.tanh(jnp.dot(h, params[0]['w']) + params[0]['b'])
        h = jnp.tanh(jnp.dot(h, params[1]['w']) + params[1]['b'])
        h = jnp.tanh(jnp.dot(h, params[2]['w']) + params[2]['b'])
        # Output Linear
        return jnp.dot(h, params[3]['w']) + params[3]['b']

    @jax.jit
    def train_batch(params, opt_state, x_batch, y_batch, lr):
        def loss_fn(p):
            logits = forward(p, x_batch)
            probs = jax.nn.sigmoid(logits)
            # MSE is used here for the specific Geodesic dynamics
            return jnp.mean((probs - y_batch) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = geodesic_update(grads, opt_state, lr)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def compute_accuracy_chunk(params, x, y):
        logits = forward(params, x)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float64)
        return jnp.sum(preds == y)

    @jax.jit
    def render_grid_inference(params, grid):
        logits = forward(params, grid)
        return jax.nn.sigmoid(logits)

    # --- DATA PREPARATION (CPU -> GPU) ---
    print("Island: Generating Spiral Data...")
    N = GeodesicConfig.N_SAMPLES
    # Use float32 for generation to save time, cast to 64 later
    theta = np.sqrt(np.random.rand(N)) * 3.0 * np.pi # 3 loops
    
    # Class A
    r_a = theta + np.random.rand(N) * 0.1
    xa = np.stack([-np.cos(theta)*r_a, np.sin(theta)*r_a], axis=1)
    
    # Class B
    r_b = theta + np.random.rand(N) * 0.1
    xb = np.stack([np.cos(theta)*r_b, -np.sin(theta)*r_b], axis=1)
    
    X_np = np.vstack((xa, xb))
    Y_np = np.vstack((np.zeros((N,1)), np.ones((N,1))))
    
    # Normalize to -1..1 range approximately
    X_np = X_np / (3.0 * np.pi + 0.5)
    
    # Move to GPU Memory once
    X_device = jax.device_put(jnp.array(X_np, dtype=jnp.float64))
    Y_device = jax.device_put(jnp.array(Y_np, dtype=jnp.float64))
    
    total_samples = 2 * N
    batch_size = GeodesicConfig.BATCH_SIZE
    steps_per_epoch = total_samples // batch_size

    # --- INITIALIZATION ---
    print("Island: Initializing Weights...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    params = [
        init_layer(keys[0], 2, 128),
        init_layer(keys[1], 128, 128),
        init_layer(keys[2], 128, 64),
        init_layer(keys[3], 64, 1)
    ]
    
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_state = GeodesicState(jnp.array(0), zeros, zeros, zeros, zeros)

    # Grid for Visualization
    res = 64
    gx = np.linspace(-1.2, 1.2, res)
    gy = np.linspace(-1.2, 1.2, res)
    xx, yy = np.meshgrid(gx, gy)
    grid_inputs = jax.device_put(jnp.array(np.c_[xx.ravel(), yy.ravel()], dtype=jnp.float64))

    # --- MAIN LOOP ---
    epoch = 0
    global_step = 0
    start_time = time.time()
    last_report_time = start_time
    
    prng_key = jax.random.PRNGKey(999)

    while not stop_event.is_set():
        epoch += 1
        
        # Shuffle Data Indices for this Epoch
        prng_key, subkey = jax.random.split(prng_key)
        perms = jax.random.permutation(subkey, total_samples)
        
        # BATCH LOOP
        for s in range(steps_per_epoch):
            # Slice Batch on GPU (View, no copy)
            idx = perms[s * batch_size : (s + 1) * batch_size]
            x_batch = X_device[idx]
            y_batch = Y_device[idx]
            
            # Train Step
            params, opt_state, loss = train_batch(params, opt_state, x_batch, y_batch, GeodesicConfig.LR)
            global_step += 1
            
            # Check Time for Visualization (Approx 15 FPS update rate)
            curr_time = time.time()
            if curr_time - last_report_time > 0.06: 
                
                # Async Evaluation (Don't do full dataset accuracy every frame, it's slow)
                # Just infer on the current batch for rough accuracy
                batch_acc = compute_accuracy_chunk(params, x_batch, y_batch) / batch_size
                
                # Render Grid
                grid_out = render_grid_inference(params, grid_inputs)
                
                # Wait for computation to finish before sending to CPU
                loss.block_until_ready()
                
                elapsed = curr_time - start_time
                tps = global_step / elapsed
                
                packet = {
                    'epoch': epoch,
                    'step': global_step,
                    'loss': float(loss),
                    'acc': float(batch_acc),
                    'grid': np.array(grid_out).reshape(res, res),
                    'tps': tps
                }
                
                # Push to GUI (Drop if full)
                try:
                    queue.put_nowait(packet)
                except mp.queues.Full:
                    pass
                
                last_report_time = curr_time
            
            if stop_event.is_set():
                break

# ==============================================================================
# 3. THE BOAT (GUI PROCESS)
# ==============================================================================
class BoatGUI:
    def __init__(self):
        self.layout = Layout()
        self.setup_layout()
        self.last_img = None
        
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
        h, w = grid_data.shape
        # Create image buffer
        img = Image.fromarray((grid_data * 255).astype('uint8'), mode='L')
        # Colorize: Black->Blue->Red->White heatmap
        img = img.convert("RGB")
        
        # Manual pixel manipulation is slow in Python, use PIL colorize if possible
        # or just simple channel swaps for speed.
        # Let's do a fast numpy tint.
        arr = np.array(img)
        # Channel 0 (R) = Value
        # Channel 2 (B) = Inverse Value
        arr[:, :, 0] = (grid_data * 255).astype(np.uint8)
        arr[:, :, 1] = 0
        arr[:, :, 2] = ((1.0 - grid_data) * 255).astype(np.uint8)
        
        img = Image.fromarray(arr)
        img = img.resize((w*2, h), resample=Image.NEAREST) # Rich pixels are roughly 1x2 aspect
        return Pixels.from_image(img)

    def update_view(self, data):
        bs = GeodesicConfig.BATCH_SIZE
        
        # Header
        self.layout["header"].update(Panel(
            f"WUBU ISLAND: [bold cyan]RTX 2080 SUPER OPTIMIZED[/bold cyan] | TPS: {data['tps']:.0f} | BATCH: {bs}", 
            style="white on black"
        ))
        
        # Telemetry
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("METRIC", style="cyan")
        table.add_column("VALUE", style="yellow", justify="right")
        
        table.add_row("EPOCH", str(data['epoch']))
        table.add_row("STEP", f"{data['step']:,}")
        table.add_row("LOSS", f"{data['loss']:.6f}")
        table.add_row("BATCH ACC", f"{data['acc']*100:.2f}%")
        
        color = "red"
        if data['acc'] > 0.90: color = "yellow"
        if data['acc'] > 0.98: color = "green"
        
        self.layout["telemetry"].update(Panel(
            table, title="[bold]NEURAL LINK[/bold]", border_style=color
        ))
        
        # Viewport
        px = self.generate_pixels(data['grid'])
        self.layout["viewport"].update(Panel(px, title="DECISION BOUNDARY", style="white on black"))
        
        return self.layout

# ==============================================================================
# 4. ORCHESTRATION
# ==============================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    # Queue size 1 ensures we only render the absolute latest frame
    comm_queue = mp.Queue(maxsize=1)
    stop_sig = mp.Event()
    
    print(">>> LAUNCHING GEODESIC ENGINE...")
    trainer = mp.Process(target=trainer_process, args=(comm_queue, stop_sig))
    trainer.start()
    
    gui = BoatGUI()
    
    try:
        with Live(gui.layout, refresh_per_second=30, screen=True) as live:
            while True:
                if stop_sig.is_set() and comm_queue.empty():
                    break
                
                try:
                    data = comm_queue.get(timeout=0.5)
                    live.update(gui.update_view(data))
                except:
                    # If queue empty (Island calculating), just yield
                    pass
                    
    except KeyboardInterrupt:
        print("\n>>> MANUAL STOP.")
        stop_sig.set()
    
    stop_sig.set()
    trainer.join()
    print(">>> SYSTEM HALTED.")