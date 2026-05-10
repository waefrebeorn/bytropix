# Wubu_Spectral_Field_v13_2_Hyperspeed_Stable.py
# THE GEOMETRIC BRAIN v13.2 [RTX 4050 HYPERSPEED STABLE]
#
# FIX: Dimensional Squeeze in Inference Step
# FEATURE: O(N) Cached Generation + Parallel Streams
# STATUS: RUNNABLE / HIGH SPEED

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
from typing import Any, List, Tuple

# --- GPU TUNING ---
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, random, lax
import optax
from flax import linen as nn
from flax.training import train_state

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
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # HARDWARE
    DTYPE = jnp.bfloat16    
    
    # GEOMETRY
    NUM_NEURONS = 2560      
    NEIGHBORS = 8          
    
    # PHYSICS
    DT = 0.1                
    SUBSTEPS = 2            
    PROP_SPEED = 4.0        
    
    # CAPACITY
    SEQ_LEN = 128           
    D_MODEL = 384           
    LAYERS = 6              
    
    # LEARNING
    LR = 4e-4
    BATCH_SIZE = 8  
    
    # GENERATION
    GEN_LENGTH = 200 
    
    VOCAB_SIZE = 8192

# ==============================================================================
# 2. GEOMETRY ENGINE
# ==============================================================================
def generate_spectral_brain():
    print(">>> GENERATING MANIFOLD...")
    
    def map_brain(p):
        d_base = np.linalg.norm(p / np.array([1.0, 0.7, 1.2]), axis=1) - 1.0
        gyri = 0.04 * np.sin(12*p[:,0]) * np.sin(12*p[:,1]) * np.sin(12*p[:,2])
        return d_base + gyri

    raw_points = (np.random.rand(200000, 3) * 3.0) - 1.5
    d_val = map_brain(raw_points)
    
    valid_mask = np.abs(d_val) < 0.08
    points = raw_points[valid_mask]
    
    if len(points) < Config.NUM_NEURONS:
        sort_idx = np.argsort(np.abs(d_val))
        points = raw_points[sort_idx][:Config.NUM_NEURONS]
    else:
        points = points[:Config.NUM_NEURONS]
        
    print(f">>> GEOMETRY LOCKED: {points.shape[0]} Neurons")

    nbrs = NearestNeighbors(n_neighbors=Config.NEIGHBORS).fit(points)
    adj_mat = nbrs.kneighbors_graph(points, mode='distance')
    
    adj_mat.data = np.exp(-adj_mat.data * 2.0)
    adj_mat = (adj_mat + adj_mat.T) / 2.0
    
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
    d_inv_sqrt[degrees == 0] = 0.0
    
    d_mat_inv = scipy.sparse.diags(d_inv_sqrt)
    laplacian = scipy.sparse.eye(Config.NUM_NEURONS) - d_mat_inv @ adj_mat @ d_mat_inv
    
    mask_in = (points[:, 2] < -0.3)
    mask_out = (points[:, 2] > 0.3)
    
    wires = [] 
    
    print(f">>> MANIFOLD READY on GPU")
    return (points, wires, 
            jnp.array(laplacian.toarray(), dtype=Config.DTYPE), 
            jnp.array(mask_in, dtype=Config.DTYPE), 
            jnp.array(mask_out, dtype=Config.DTYPE))

POINTS_3D, WIRES, LAPLACIAN_GPU, MASK_IN, MASK_OUT = generate_spectral_brain()

# ==============================================================================
# 3. PHYSICS & MODEL
# ==============================================================================

class SpectralLayer(nn.Module):
    config: Any 
    
    @nn.compact
    def __call__(self, state, forcing):
        u, v = state
        forcing = forcing.astype(Config.DTYPE)
        drive = nn.Dense(1, dtype=Config.DTYPE)(forcing) 
        gamma = self.param('gamma', nn.initializers.constant(0.1), (1,))
        
        def micro_step(carry, _):
            curr_u, curr_v = carry
            # [B, N, 1]
            lap_u = jnp.einsum('nm,bmc->bnc', LAPLACIAN_GPU, curr_u)
            accel = -Config.PROP_SPEED * lap_u - gamma * curr_v + drive
            new_v = curr_v + accel * Config.DT
            new_u = curr_u + new_v * Config.DT
            new_u = jnp.tanh(new_u) 
            new_u = new_u.astype(Config.DTYPE)
            new_v = new_v.astype(Config.DTYPE)
            return (new_u, new_v), None

        (final_u, final_v), _ = lax.scan(micro_step, (u, v), None, length=Config.SUBSTEPS)
        return (final_u, final_v), final_u

class GeodesicBrain(nn.Module):
    vocab_size: int

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, Config.D_MODEL, dtype=Config.DTYPE)
        self.sensory_proj = nn.Dense(Config.NUM_NEURONS, dtype=Config.DTYPE)
        self.output_proj = nn.Dense(self.vocab_size, dtype=Config.DTYPE)
        
        self.layers = [SpectralLayer(Config, name=f'spec_{i}') for i in range(Config.LAYERS)]
        self.norms = [nn.LayerNorm(dtype=Config.DTYPE, name=f'norm_{i}') for i in range(Config.LAYERS)]

    def __call__(self, x, carry=None):
        # TRAINING MODE (Scan over time)
        B = x.shape[0]
        if carry is None:
            init_u = jnp.zeros((B, Config.NUM_NEURONS, 1), dtype=Config.DTYPE)
            init_v = jnp.zeros((B, Config.NUM_NEURONS, 1), dtype=Config.DTYPE)
            carry = tuple([(init_u, init_v) for _ in range(Config.LAYERS)])
            
        x_emb = self.embed(x)
        sensory = self.sensory_proj(x_emb)
        sensory = sensory * MASK_IN[None, None, :] 
        sensory = sensory[..., None] 
        
        def physics_scan(carry, input_slice):
            new_carry = []
            layer_in = input_slice
            
            for i, layer in enumerate(self.layers):
                layer_state = carry[i]
                (new_u, new_v), wave_out = layer(layer_state, layer_in)
                new_carry.append((new_u, new_v))
                
                resid = layer_in + wave_out
                resid_flat = resid.squeeze(-1)
                resid_norm = self.norms[i](resid_flat)
                layer_in = resid_norm[..., None]
            
            return tuple(new_carry), layer_in

        ScanRNN = nn.scan(
            lambda m, c, i: physics_scan(c, i),
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1
        )
        
        final_carry, history = ScanRNN(self, carry, sensory)
        
        output_activity = history.squeeze(-1) * MASK_OUT[None, None, :]
        logits = self.output_proj(output_activity)
        logits = logits.astype(jnp.float32)
        
        return logits, history, final_carry

    # INFERENCE MODE: Single Step (No Scan)
    def step_forward(self, carry, x_token):
        # x_token: [Batch, 1]
        x_emb = self.embed(x_token) 
        # FIX: Squeeze time dimension [B, 1, D] -> [B, D]
        x_emb = x_emb.squeeze(1) 
        
        sensory = self.sensory_proj(x_emb) 
        # [B, N] mask broadcast
        sensory = sensory * MASK_IN[None, :] 
        layer_in = sensory[..., None] # [B, N, 1]
        
        new_carry = []
        
        for i, layer in enumerate(self.layers):
            layer_state = carry[i]
            # Now inputs match [B, N, 1] so no broadcasting explosion
            (new_u, new_v), wave_out = layer(layer_state, layer_in)
            new_carry.append((new_u, new_v))
            
            resid = layer_in + wave_out
            resid_flat = resid.squeeze(-1)
            resid_norm = self.norms[i](resid_flat)
            layer_in = resid_norm[..., None]
        
        output_activity = layer_in.squeeze(-1) * MASK_OUT[None, :] 
        logits = self.output_proj(output_activity) # [B, Vocab]
        
        # Restore time dim for generator compatibility
        logits = logits[:, None, :] # [B, 1, Vocab]
        logits = logits.astype(jnp.float32)
        
        return logits, tuple(new_carry)

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train_process(queue, stop_event):
    try:
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Wubu " * 10000
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=Config.VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        
        data_np = np.array(tokenizer.encode(text).ids, dtype=np.int32)
        data = jax.device_put(jnp.array(data_np))
        
        rng = random.PRNGKey(42)
        model = GeodesicBrain(vocab_size=Config.VOCAB_SIZE)
        
        # Initialize
        dummy_x = jnp.zeros((Config.BATCH_SIZE, Config.SEQ_LEN), dtype=jnp.int32)
        variables = model.init(rng, dummy_x)
        params = variables['params']
        
        tx = optax.adamw(Config.LR)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @jit
        def train_step(state, batch):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            def loss_fn(params):
                logits, activity, _ = state.apply_fn({'params': params}, inputs)
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
                energy = jnp.mean(activity ** 2)
                return loss + 0.001 * energy, (loss, activity)
            
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, (raw_loss, act)), grads = grad_fn(state.params)
            return state.apply_gradients(grads=grads), loss, act

        # --- HYPERSPEED GENERATOR ---
        @jit
        def fast_generate(params, rng, prompt):
            # prompt: [Batch, SeqLen]
            # 1. Warmup
            _, _, carry = model.apply({'params': params}, prompt)
            
            # Start with last token of prompt
            curr_token = prompt[:, -1:]
            
            def gen_step(i, carry_and_token):
                carry, token, key = carry_and_token
                key, sk = random.split(key)
                
                logits, new_carry = model.apply({'params': params}, carry, token, method=GeodesicBrain.step_forward)
                
                next_logit = logits[:, 0, :] / 0.7 
                next_idx = random.categorical(sk, next_logit)
                next_token = next_idx[:, None]
                
                return (new_carry, next_token, key)

            # 2. Fast Forward Loop
            def scan_gen(carry_and_token, _):
                new_state = gen_step(0, carry_and_token)
                return new_state, new_state[1]

            final_state, tokens = lax.scan(scan_gen, (carry, curr_token, rng), None, length=Config.GEN_LENGTH)
            
            return tokens.squeeze(-1) 

        queue.put({'type': 'READY'})
        step = 0
        t0 = time.time()
        
        while not stop_event.is_set():
            rng, k = random.split(rng)
            
            # Get Batch
            idxs = random.randint(k, (Config.BATCH_SIZE,), 0, len(data) - Config.SEQ_LEN - 1)
            batch = jnp.stack([lax.dynamic_slice(data, (i,), (Config.SEQ_LEN + 1,)) for i in idxs])
            
            # Train
            state, loss_val, activity = train_step(state, batch)
            
            step += 1
            
            if step % 25 == 0:
                # Generate from the training batch prompt
                prompt = batch[:, :32] # Use first 32 tokens as prompt
                gen_rng, _ = random.split(rng)
                
                t_gen_start = time.time()
                # Generates 8 streams simultaneously
                gen_tokens = fast_generate(state.params, gen_rng, prompt)
                gen_tokens.block_until_ready()
                t_gen_end = time.time()
                
                # Decode all 8 streams
                gen_np = np.array(gen_tokens)
                texts = [tokenizer.decode(gen_np[i]) for i in range(Config.BATCH_SIZE)]
                
                gen_speed = (Config.BATCH_SIZE * Config.GEN_LENGTH) / (t_gen_end - t_gen_start + 1e-9)
                
                t1 = time.time()
                train_speed = (Config.BATCH_SIZE * Config.SEQ_LEN * 25) / (t1 - t0 + 1e-9)
                t0 = t1
                
                vis_act = np.array(activity[0, -1, :, 0], dtype=np.float32)
                
                msg = {
                    'step': step,
                    'loss': float(loss_val),
                    'train_tps': train_speed,
                    'gen_tps': gen_speed,
                    'texts': texts, # List of 8 texts
                    'activity': vis_act
                }
                try: queue.put_nowait({'type': 'DATA', 'data': msg})
                except: pass
            
            elif step % 2 == 0:
                vis_act = np.array(activity[0, -1, :, 0], dtype=np.float32)
                try: queue.put_nowait({'type': 'LITE', 'loss': float(loss_val), 'activity': vis_act})
                except: pass

    except Exception as e:
        err_msg = traceback.format_exc()
        queue.put({'type': 'ERROR', 'data': err_msg})

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
class RTXGUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(Layout(name="H", size=3), Layout(name="B", ratio=1), Layout(name="F", size=3))
        self.layout["B"].split_row(Layout(name="VIS", ratio=2), Layout(name="STAT", ratio=1))
        
        self.loss_history = []
        self.console = Console()
        self.angle = 0.0
        self.train_tps = 0.0
        self.gen_tps = 0.0
        self.texts = ["Initializing Hyperspeed Matrix..."] * 8
        
        self.points = POINTS_3D
        self.mask_in = np.array(MASK_IN)
        self.mask_out = np.array(MASK_OUT)
        
    def render(self, d, lite=False):
        term_w, term_h = shutil.get_terminal_size()
        
        if not lite:
            self.texts = d['texts']
            self.train_tps = d['train_tps']
            self.gen_tps = d['gen_tps']
        
        self.loss_history.append(d['loss'])
        if len(self.loss_history) > 100: self.loss_history.pop(0)

        # FAST RENDER
        w, h = int(term_w * 0.6), (term_h - 8) * 2
        img = Image.new('RGB', (w, h), (10, 10, 15))
        draw = ImageDraw.Draw(img)
        
        cx, cy = w/2, h/2
        scale = min(w, h) * 0.35
        self.angle += 0.08 
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        
        act = d['activity']
        act = np.tanh(act * 2.0)
        
        pts_x = self.points[:, 0]
        pts_y = self.points[:, 1]
        pts_z = self.points[:, 2]
        
        rx = pts_x * ca - pts_z * sa
        rz = pts_x * sa + pts_z * ca
        
        cols = np.zeros((len(act), 3))
        cols[:] = (50, 50, 90) 
        cols[self.mask_in > 0] = (40, 150, 40)
        cols[self.mask_out > 0] = (150, 40, 40)
        
        glow = np.clip(act, 0, 1)[:, None]
        final_cols = cols * (1 - glow) + np.array([255, 255, 200]) * glow
        
        depth_order = np.argsort(rz)
        for idx in depth_order[::2]:
            px = cx + rx[idx] * scale
            py = cy + pts_y[idx] * scale
            c = tuple(final_cols[idx].astype(int))
            draw.point((px, py), fill=c)
            draw.point((px+1, py), fill=c)
            draw.point((px, py+1), fill=c)

        # CHART
        w_t = term_w - w - 8
        img_chart = Image.new('RGB', (w_t, h), (0,0,0))
        draw_c = ImageDraw.Draw(img_chart)
        if len(self.loss_history) > 1:
            min_l, max_l = min(self.loss_history), max(self.loss_history)
            rng = max_l - min_l + 1e-6
            pts = []
            for i, val in enumerate(self.loss_history):
                x = int((i/100) * w_t)
                y = int(h - ((val - min_l)/rng * (h-10)) - 5)
                pts.append((x,y))
            draw_c.line(pts, fill=(0, 255, 0), width=2)

        # Format 8 streams of text
        text_panel_content = ""
        for i, txt in enumerate(self.texts[:4]): # Show first 4
            text_panel_content += f"[bold cyan]CH{i}:[/] {txt[-100:].replace(chr(10), ' ')}\n"

        stat_txt = f"""
        [bold green]RTX 4050 HYPERSPEED[/]
        ------------------
        Batch:     {Config.BATCH_SIZE}
        Loss:      {d['loss']:.4f}
        Train TPS: {self.train_tps:.1f}
        [bold yellow]GEN TPS:   {self.gen_tps:.1f}[/]
        
        [dim]Parallel Output (4/{Config.BATCH_SIZE})[/]
        """
        
        self.layout["H"].update(Panel("WUBU v13.2 [STABLE]", style="white on blue"))
        self.layout["VIS"].update(Panel(Pixels.from_image(img), title="MANIFOLD"))
        self.layout["STAT"].update(Panel(stat_txt + "\n" + text_panel_content, title="STATS & OUTPUT"))
        self.layout["F"].update(Panel(f"[dim]Loss Graph (Min: {min(self.loss_history):.3f})[/]", title_align="right"))
        self.layout["STAT"].split_column(Panel(stat_txt + "\n" + text_panel_content), Panel(Pixels.from_image(img_chart)))
        
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    q = mp.Queue()
    stop = mp.Event()
    
    p = mp.Process(target=train_process, args=(q, stop))
    p.start()
    
    gui = RTXGUI()
    console = Console()
    
    try:
        with console.status("[bold green]Compiling Hyperspeed Kernels...[/]") as status:
            while True:
                try:
                    msg = q.get(timeout=1)
                    if msg['type'] == 'READY': break
                    if msg['type'] == 'ERROR': 
                        stop.set()
                        p.terminate()
                        console.print(Panel(f"[red bold]STARTUP ERROR:[/]\n{msg['data']}"))
                        sys.exit(1)
                except: pass
        
        with Live(gui.layout, refresh_per_second=20, screen=True) as live:
            while p.is_alive():
                try:
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA':
                        live.update(gui.render(msg['data'], lite=False))
                    elif msg['type'] == 'LITE':
                        live.update(gui.render(msg, lite=True))
                    elif msg['type'] == 'ERROR':
                        live.stop()
                        console.print(Panel(f"[red bold]RUNTIME ERROR:[/]\n{msg['data']}"))
                        stop.set()
                        break
                except: 
                    time.sleep(0.005)
                    
    except KeyboardInterrupt:
        stop.set()
    finally:
        p.terminate()
        print("System Halted.")