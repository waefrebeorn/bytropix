import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import struct
import optax
import numpy as np
import cv2
import time
import threading
import pickle
import sys
import signal
import subprocess
import scipy.io.wavfile as wav
from pathlib import Path
from functools import partial
from collections import deque
import argparse
from typing import Any

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console, Group
from rich.align import Align
from rich.text import Text
from PIL import Image
from dataclasses import dataclass

try:
    from rich_pixels import Pixels
except ImportError:
    Pixels = None

# --- VHF CANVAS CONSTANTS (Audio Ready) ---
VBI_LINES = 45
VISIBLE_H = 480
CANVAS_H = VISIBLE_H + VBI_LINES

# We add 16 "pixels" to the start of every line for the Audio/FM Lock-On
AUDIO_HBI_WIDTH = 16 
VISIBLE_W = 640
CANVAS_W = AUDIO_HBI_WIDTH + VISIBLE_W 

TOTAL_PIXELS = CANVAS_H * CANVAS_W

# --- INTERACTIVITY ---
class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock()
        self.force_save = False
        self.shutdown_event = threading.Event()
    def get_and_reset_force_save(self):
        with self.lock:
            save = self.force_save
            self.force_save = False
            return save
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    print("--- Controls: [s] Force Save | [q] Quit ---")
    import select, sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not shared_state.shutdown_event.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                with shared_state.lock:
                    if key in ['q', '\x03']: shared_state.set_shutdown(); break
                    elif key == 's': shared_state.force_save = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# --- Q-CONTROLLER ---
class QControllerState(struct.PyTreeNode):
    q_table: jax.Array
    metric_history: jax.Array
    current_lr: jax.Array
    exploration_rate: jax.Array
    step_count: jax.Array
    last_action_idx: jax.Array
    status_code: jax.Array

@dataclass(frozen=True)
class QControllerConfig:
    num_lr_actions: int = 5
    lr_change_factors: tuple = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    metric_history_len: int = 100
    exploration_rate_q: float = 0.3
    min_exploration_rate: float = 0.05
    exploration_decay: float = 0.9998
    warmup_steps: int = 500
    warmup_lr_start: float = 1e-6

def init_q_controller(config: QControllerConfig) -> QControllerState:
    return QControllerState(
        q_table=jnp.zeros(config.num_lr_actions),
        metric_history=jnp.zeros(config.metric_history_len),
        current_lr=jnp.array(config.warmup_lr_start),
        exploration_rate=jnp.array(config.exploration_rate_q),
        step_count=jnp.array(0),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        status_code=jnp.array(0, dtype=jnp.int32)
    )

@partial(jax.jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state: QControllerState, key: jax.Array, config: QControllerConfig, target_lr: float) -> QControllerState:
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0, dtype=jnp.int32))
    def regular_action():
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < jnp.squeeze(state.exploration_rate),
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions, dtype=jnp.int32),
            lambda: jnp.argmax(state.q_table).astype(jnp.int32))
        new_lr = jnp.clip(state.current_lr * jnp.array(config.lr_change_factors)[action_idx], config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx, status_code=jnp.array(0, dtype=jnp.int32))
    return jax.lax.cond(jnp.squeeze(state.step_count) < config.warmup_steps, warmup_action, regular_action)

@partial(jax.jit, static_argnames=('config',))
def q_controller_update(state: QControllerState, metric_value: jax.Array, config: QControllerConfig) -> QControllerState:
    new_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    st = state.replace(metric_history=new_history)
    def perform_q_update(s):
        reward = -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 10, 10))
        is_improving = reward > -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 20, 10))
        status = jax.lax.select(is_improving, jnp.array(1, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32))
        old_q = s.q_table[s.last_action_idx]
        new_q = old_q + config.learning_rate_q * (reward - old_q)
        return s.replace(q_table=s.q_table.at[s.last_action_idx].set(new_q), exploration_rate=jnp.maximum(config.min_exploration_rate, s.exploration_rate * config.exploration_decay), status_code=status)
    can_update = (jnp.squeeze(st.step_count) > config.warmup_steps) & (jnp.squeeze(st.last_action_idx) >= 0)
    return jax.lax.cond(can_update, perform_q_update, lambda s: s, st)

# --- MATH HELPERS ---
def rgb_to_hsl_jax(rgb):
    epsilon = 1e-8
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = jnp.maximum(jnp.maximum(r, g), b)
    cmin = jnp.minimum(jnp.minimum(r, g), b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0
    s = jnp.where(delta < epsilon, 0.0, delta / (1.0 - jnp.abs(2.0 * l - 1.0) + epsilon))
    h = jnp.zeros_like(r)
    h = jnp.where(cmax == r, (((g - b) / (delta + epsilon))) % 6.0, h)
    h = jnp.where(cmax == g, (((b - r) / (delta + epsilon)) + 2.0), h)
    h = jnp.where(cmax == b, (((r - g) / (delta + epsilon)) + 4.0), h)
    h = h / 6.0
    h = jnp.where(delta < epsilon, 0.0, h)
    return jnp.stack([h, s, l], axis=-1)

def circular_l1_loss(pred, target):
    diff = jnp.abs(pred - target)
    return jnp.minimum(diff, 1.0 - diff)

# --- MODEL ARCHITECTURE ---
class CustomTrainState(train_state.TrainState):
    ema_params: Any
    q_controller_state: QControllerState

class PositionalEncoding(nn.Module):
    num_freqs: int = 10
    
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs) * jnp.pi
        features = [x]
        for freq in freqs:
            features.append(jnp.sin(x * freq))
            features.append(jnp.cos(x * freq))
        return jnp.concatenate(features, axis=-1)

class HamiltonEncoder(nn.Module):
    latent_grid_size: int = 96
    d_model: int = 512
    
    @nn.compact
    def __call__(self, images_rgb):
        x = images_rgb
        features = 32
        current_h, current_w = x.shape[1], x.shape[2]
        context_vectors = []
        
        while (current_h // 2) >= self.latent_grid_size and (current_w // 2) >= self.latent_grid_size:
            x = nn.Conv(features, (4, 4), strides=(2, 2))(x)
            x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2
            current_h //= 2
            current_w //= 2
            
        if context_vectors:
            context_vector = jnp.concatenate(context_vectors, axis=-1)
        else:
            context_vector = jnp.zeros((x.shape[0], 1))
            
        if x.shape[1] != self.latent_grid_size or x.shape[2] != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
            
        x = nn.Conv(self.d_model, (3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        
        raw_params = nn.Conv(5, (1, 1))(x)
        quat_raw = raw_params[..., :4]
        quaternions = quat_raw / (jnp.linalg.norm(quat_raw, axis=-1, keepdims=True) + 1e-6)
        amplitude = nn.sigmoid(raw_params[..., 4:5])
        
        return jnp.concatenate([quaternions, amplitude], axis=-1), context_vector

class VHFDecoder(nn.Module):
    d_model: int = 512
    
    @nn.compact
    def __call__(self, hamilton_keys, context_vector, coords):
        B, H, W, C = hamilton_keys.shape
        
        y_rescaled = (coords[..., 1] + 1.0) / 2.0 * (H - 1)
        x_rescaled = (coords[..., 0] + 1.0) / 2.0 * (W - 1)
        coords_yx = jnp.stack([y_rescaled, x_rescaled], axis=-1)
        
        def sample_one_image(grid, c_yx):
            grid_chw = grid.transpose(2, 0, 1)
            return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, c_yx.T, order=1, mode='nearest'))(grid_chw).T
            
        local_features = jax.vmap(sample_one_image)(hamilton_keys, coords_yx)
        
        encoded_coords = PositionalEncoding(num_freqs=10)(coords)
        
        context_tiled = jnp.repeat(context_vector[:, None, :], coords.shape[1], axis=1)
        
        h = jnp.concatenate([encoded_coords, context_tiled, local_features], axis=-1)
        
        for _ in range(4): 
            h = nn.gelu(nn.Dense(self.d_model)(h))
            
        return nn.tanh(nn.Dense(3)(h))

class VHFEndToEndModel(nn.Module):
    latent_grid_size: int = 96
    d_model: int = 512
    
    def setup(self):
        self.encoder = HamiltonEncoder(latent_grid_size=self.latent_grid_size, d_model=self.d_model)
        self.decoder = VHFDecoder(d_model=self.d_model)

    def __call__(self, images_rgb, coords):
        keys, ctx = self.encoder(images_rgb)
        return self.decoder(keys, ctx, coords), keys, ctx

    def encode(self, images_rgb):
        return self.encoder(images_rgb)

    def decode(self, hamilton_keys, context_vector, coords):
        return self.decoder(hamilton_keys, context_vector, coords)

# --- DATA PIPELINE (Native Audio Injection) ---
def extract_audio_from_video(video_path):
    temp_wav = "temp_train_audio.wav"
    subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', temp_wav, '-y'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        return temp_wav
    return None

def video_audio_generator(video_path, audio_path, batch_size):
    if audio_path is None or not os.path.exists(audio_path):
        print("[*] No audio file provided. Extracting natively via ffmpeg...")
        audio_path = extract_audio_from_video(video_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    
    if audio_path and os.path.exists(audio_path):
        sr, full_audio = wav.read(audio_path)
    else:
        sr, full_audio = 44100, np.zeros(44100 * 60)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 24.0
    samples_per_frame = int(sr / fps)
    
    batch_frames, batch_audio = [], []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
            
        # 1. Video Frame Processing
        frame = cv2.resize(frame, (VISIBLE_W, VISIBLE_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_norm = (frame.astype(np.float32) / 127.5) - 1.0
        
        # 2. Audio Processing (Synchronized Chunking)
        start_idx = frame_idx * samples_per_frame
        end_idx = start_idx + samples_per_frame
        
        if start_idx < len(full_audio):
            audio_chunk = full_audio[start_idx:end_idx]
            if len(audio_chunk) < samples_per_frame:
                audio_chunk = np.pad(audio_chunk, (0, samples_per_frame - len(audio_chunk)))
        else:
            audio_chunk = np.zeros(samples_per_frame)
        
        # Normalize audio to [-1.0, 1.0] to match video tensors
        if audio_chunk.max() > audio_chunk.min():
            audio_norm = 2.0 * ((audio_chunk - audio_chunk.min()) / (audio_chunk.max() - audio_chunk.min())) - 1.0
        else:
            audio_norm = np.zeros_like(audio_chunk, dtype=np.float32)
            
        # Pad and reshape into the 2D HBI Strip [CANVAS_H, AUDIO_HBI_WIDTH, 3]
        target_size = CANVAS_H * AUDIO_HBI_WIDTH
        if len(audio_norm) < target_size:
            padded_audio = np.pad(audio_norm, (0, target_size - len(audio_norm)), 'constant')
        else:
            padded_audio = audio_norm[:target_size]
            
        audio_strip = padded_audio.reshape((CANVAS_H, AUDIO_HBI_WIDTH))
        audio_strip_rgb = np.stack([audio_strip]*3, axis=-1)
        
        batch_frames.append(frame_norm)
        batch_audio.append(audio_strip_rgb)
        frame_idx += 1
        
        if len(batch_frames) == batch_size: 
            yield np.array(batch_frames), np.array(batch_audio)
            batch_frames, batch_audio = [], []

# --- TRAINER ---
class VideoVHFTrainer:
    def __init__(self, args):
        self.args = args
        self.model = VHFEndToEndModel(latent_grid_size=args.latent_grid_size, d_model=args.d_model)
        self.data_gen = video_audio_generator(args.video, args.audio, args.batch_size)
        self.interactive = InteractivityState()
        self.q_config = QControllerConfig(warmup_lr_start=1e-6, warmup_steps=500)
        signal.signal(signal.SIGINT, lambda s,f: self.interactive.set_shutdown())
        
        self.ui_lock = threading.Lock()
        self.loss_hist = deque(maxlen=100)
        self.q_lr = 0.0
        self.q_status = 0
        self.steps_per_sec = 0.0
        self.last_metrics = {'composite_loss': 0.0, 'luma_loss': 0.0, 'phase_loss': 0.0, 'sat_loss': 0.0}
        self.ui_state = {'gt': Text("Initializing Camera...", justify="center"), 
                         'latent': Text("Waking up Hamilton Space...", justify="center"), 
                         'pred': Text("Warming up Electron Beam...", justify="center")}

    def _save_checkpoint(self, state, step, path):
        data = {
            'global_step': step,
            'params': jax.device_get(state.params),
            'ema_params': jax.device_get(state.ema_params),
            'opt_state': jax.device_get(state.opt_state),
            'q_controller_state': jax.device_get(state.q_controller_state)
        }
        with open(path, 'wb') as f: pickle.dump(data, f)

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            layout = Layout()
            header = Panel(Align.center(f"[bold cyan]📺 VHF Hamilton Modulator + Audio[/] | Step: [bold yellow]{self.step_count}[/] | Base Resolution: {CANVAS_W}x{CANVAS_H} (VBI+HBI)"), style="bold magenta")
            
            stats_table = Table.grid(expand=True, padding=(0,2))
            stats_table.add_column(justify="left")
            stats_table.add_column(justify="left")
            stats_table.add_column(justify="left")
            stats_table.add_column(justify="left")
            
            q_status_str = {0: "[cyan]Warmup[/]", 1: "[green]Improving[/]", 2: "[yellow]Stagnated[/]"}.get(self.q_status, "Unknown")
            
            stats_table.add_row(
                f"🚀 [bold]FPS:[/] {self.steps_per_sec:.2f} sweeps/sec",
                f"📉 [bold]Comp Loss:[/] [bold green]{self.last_metrics['composite_loss']:.5f}[/] (Luma: {self.last_metrics['luma_loss']:.4f} | Phase: {self.last_metrics['phase_loss']:.4f})",
                f"🎛️ [bold]Q-State:[/] {q_status_str} (LR: {self.q_lr:.2e})",
                f"📐 [bold]Grid/Capacity:[/] {self.args.latent_grid_size}x{self.args.latent_grid_size} | D-Model: {self.args.d_model}"
            )
            stats_panel = Panel(stats_table, title="[bold yellow]📡 Real-Time Telemetry[/]", border_style="yellow")

            spark_chars = " ▂▃▄▅▆▇█"
            if len(self.loss_hist) > 2:
                hist = np.array(self.loss_hist)
                bins = np.linspace(hist.min(), hist.max(), len(spark_chars))
                indices = np.clip(np.digitize(hist, bins) - 1, 0, len(spark_chars) - 1)
                spark_str = "".join(spark_chars[i] for i in indices)
            else:
                spark_str = "Gathering data..."
            spark_panel = Panel(Align.center(f"[cyan]{spark_str}[/]"), title="1D Raster Sweep Loss History", border_style="cyan")
            
            top_group = Group(stats_panel, spark_panel)

            vid_table = Table.grid(expand=True, padding=(0,1))
            vid_table.add_column(ratio=1); vid_table.add_column(ratio=1); vid_table.add_column(ratio=1)
            vid_table.add_row(
                Panel(self.ui_state['gt'], title="1. Original Canvas (VBI Top, Audio Left) 📸", border_style="dim"),
                Panel(self.ui_state['latent'], title="2. Hamilton Key Space 🔑", border_style="blue"),
                Panel(self.ui_state['pred'], title="3. Reconstructed Canvas ✨", border_style="green")
            )
            
            layout.split(Layout(header, size=3), Layout(top_group, size=8), Layout(vid_table, ratio=1))
            return layout

    @partial(jax.jit, static_argnames=('self', 'pixels_per_step'))
    def train_step(self, state, visible_frames, audio_strips, key, pixels_per_step):
        B = visible_frames.shape[0]
        q_rng, loss_rng, t_rng = jax.random.split(key, 3)

        def loss_fn(params):
            hamilton_keys, context_vector = self.model.apply({'params': params}, visible_frames, method=self.model.encode)
            start_t = jax.random.randint(t_rng, (B,), 0, TOTAL_PIXELS - pixels_per_step)
            offsets = jnp.arange(pixels_per_step, dtype=jnp.float32)
            sampled_t = start_t[:, None] + offsets[None, :]
            
            x_coords = 2.0 * ((sampled_t % CANVAS_W) / (CANVAS_W - 1)) - 1.0
            y_coords = 2.0 * (jnp.floor(sampled_t / CANVAS_W) / (CANVAS_H - 1)) - 1.0
            coords = jnp.stack([x_coords, y_coords], axis=-1)
            
            pred_rgb = self.model.apply({'params': params}, hamilton_keys, context_vector, coords, method=self.model.decode)
            
            is_vbi = (sampled_t < (VBI_LINES * CANVAS_W))[..., None]
            is_hbi_audio = ((sampled_t % CANVAS_W) < AUDIO_HBI_WIDTH)[..., None]
            
            # The VBI Target
            context_padded = jnp.pad(context_vector, ((0,0), (0, max(0, 3 - context_vector.shape[1]))))[:, :3]
            vbi_target = jnp.repeat(context_padded[:, None, :], pixels_per_step, axis=1)

            # The VISIBLE RGB Target
            visible_y_idx = jnp.clip(jnp.floor(sampled_t / CANVAS_W) - VBI_LINES, 0, VISIBLE_H - 1).astype(jnp.int32)
            visible_x_idx = jnp.clip((sampled_t % CANVAS_W) - AUDIO_HBI_WIDTH, 0, VISIBLE_W - 1).astype(jnp.int32)
            gt_visible = visible_frames[jnp.arange(B)[:, None], visible_y_idx, visible_x_idx, :]
            
            # The HBI Target (The REAL Audio Waveform)
            audio_y_idx = jnp.clip(jnp.floor(sampled_t / CANVAS_W), 0, CANVAS_H - 1).astype(jnp.int32)
            audio_x_idx = jnp.clip((sampled_t % CANVAS_W), 0, AUDIO_HBI_WIDTH - 1).astype(jnp.int32)
            audio_target = audio_strips[jnp.arange(B)[:, None], audio_y_idx, audio_x_idx, :]
            
            gt_canvas = jnp.where(is_vbi, vbi_target, jnp.where(is_hbi_audio, audio_target, gt_visible))
            
            # --- THE POINCARÉ / HSL MANIFOLD FIX ---
            pred_rgb_norm = jnp.clip((pred_rgb + 1.0) / 2.0, 0.0, 1.0)
            gt_canvas_norm = jnp.clip((gt_canvas + 1.0) / 2.0, 0.0, 1.0)
            
            pred_hsl = rgb_to_hsl_jax(pred_rgb_norm)
            gt_hsl = rgb_to_hsl_jax(gt_canvas_norm)
            
            loss_h = circular_l1_loss(pred_hsl[..., 0], gt_hsl[..., 0])
            loss_s = jnp.abs(pred_hsl[..., 1] - gt_hsl[..., 1])
            loss_l = jnp.abs(pred_hsl[..., 2] - gt_hsl[..., 2])
            
            LUMA_WEIGHT = 10.0
            PHASE_WEIGHT = 2.0
            SAT_WEIGHT = 1.0
            
            loss_h_mean = jnp.mean(loss_h)
            loss_s_mean = jnp.mean(loss_s)
            loss_l_mean = jnp.mean(loss_l)
            
            composite_loss = (LUMA_WEIGHT * loss_l_mean) + (PHASE_WEIGHT * loss_h_mean) + (SAT_WEIGHT * loss_s_mean)
            
            metrics = {
                'composite_loss': composite_loss,
                'luma_loss': loss_l_mean,
                'phase_loss': loss_h_mean,
                'sat_loss': loss_s_mean
            }
            
            return composite_loss, (pred_rgb, hamilton_keys, gt_canvas, metrics)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = aux[3]
        
        new_q_pre = q_controller_choose_action(state.q_controller_state, q_rng, self.q_config, self.args.lr)
        current_lr = new_q_pre.current_lr
        
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=current_lr)
        new_params = optax.apply_updates(state.params, updates)
        new_ema = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * 0.001, state.ema_params, new_params)
        
        final_q = q_controller_update(new_q_pre, metrics['composite_loss'], self.q_config)
        new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema, q_controller_state=final_q)
        
        return new_state, metrics, current_lr, final_q.status_code

    @partial(jax.jit, static_argnames=('self',))
    def eval_canvases(self, params, visible_frame, audio_strip):
        keys, ctx = self.model.apply({'params': params}, visible_frame[None, ...], method=self.model.encode)
        
        t = jnp.arange(TOTAL_PIXELS, dtype=jnp.float32)
        x_coords = 2.0 * ((t % CANVAS_W) / (CANVAS_W - 1)) - 1.0
        y_coords = 2.0 * (jnp.floor(t / CANVAS_W) / (CANVAS_H - 1)) - 1.0
        coords = jnp.stack([x_coords, y_coords], axis=-1)[None, ...]
        pred_rgb = self.model.apply({'params': params}, keys, ctx, coords, method=self.model.decode)
        
        ctx_pad = jnp.pad(ctx[0], ((0, max(0, 3 - ctx.shape[1]))))[:3]
        vbi_block = jnp.tile(ctx_pad, (VBI_LINES, CANVAS_W, 1))
        
        audio_hbi_visible_region = audio_strip[VBI_LINES:, :AUDIO_HBI_WIDTH, :]
        video_row = jnp.concatenate([audio_hbi_visible_region, visible_frame], axis=1)
        
        gt_canvas = jnp.concatenate([vbi_block, video_row], axis=0)
        
        return gt_canvas, keys[0], pred_rgb.reshape((CANVAS_H, CANVAS_W, 3))

    def train(self):
        Console().print("[bold green]--- Booting VHF Phase 1 Trainer (Audio FM + HSL Manifold) ---[/]")
        threading.Thread(target=listen_for_keys, args=(self.interactive,), daemon=True).start()
        
        key = jax.random.PRNGKey(self.args.seed)
        dummy_img = jnp.zeros((self.args.batch_size, VISIBLE_H, VISIBLE_W, 3))
        dummy_coords = jnp.zeros((self.args.batch_size, 1024, 2))
        params = self.model.init(key, dummy_img, dummy_coords)['params']
        
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr))
        q_state = init_q_controller(self.q_config)
        
        ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_{self.args.latent_grid_size}L_vhf.pkl")
        self.step_count = 0
        
        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            params = data['params']
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=tx, ema_params=data['ema_params'], q_controller_state=data['q_controller_state'])
            state = state.replace(opt_state=data['opt_state'], step=data['global_step'])
            self.step_count = data['global_step']
        else:
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=tx, ema_params=params, q_controller_state=q_state)
        
        live = Live(self._generate_layout(), refresh_per_second=10)
        last_time = time.time()
        
        with live:
            for visible_frames, audio_strips in self.data_gen:
                if self.interactive.shutdown_event.is_set(): break
                
                key, subkey = jax.random.split(key)
                state, metrics, current_lr, q_status = self.train_step(state, visible_frames, audio_strips, subkey, self.args.pixels_per_step)
                self.step_count += 1
                
                curr_time = time.time()
                current_fps = 1.0 / (curr_time - last_time + 1e-6)
                last_time = curr_time
                
                with self.ui_lock:
                    metrics_np = jax.device_get(metrics)
                    self.last_metrics = metrics_np
                    self.loss_hist.append(float(metrics_np['composite_loss']))
                    self.q_lr = float(current_lr)
                    self.q_status = int(q_status)
                    self.steps_per_sec = current_fps
                
                if self.step_count % 20 == 0 and Pixels:
                    gt_rgb, hamilton_keys, pred_rgb = self.eval_canvases(state.ema_params, visible_frames[0], audio_strips[0])
                    latent_resized = jax.image.resize(hamilton_keys[..., :3], (CANVAS_H, CANVAS_W, 3), method='nearest')
                    
                    term_w = 75 
                    term_h = int(term_w * (CANVAS_H / CANVAS_W) * 0.5)
                    
                    def to_rich(img_array, smooth=True):
                        arr_np = np.array(jnp.clip(img_array * 0.5 + 0.5, 0, 1) * 255, dtype=np.uint8)
                        method = Image.LANCZOS if smooth else Image.NEAREST
                        return Pixels.from_image(Image.fromarray(arr_np).resize((term_w, term_h), method))
                    
                    with self.ui_lock:
                        self.ui_state['gt'] = Align.center(to_rich(gt_rgb))
                        self.ui_state['latent'] = Align.center(to_rich(latent_resized, smooth=False))
                        self.ui_state['pred'] = Align.center(to_rich(pred_rgb))
                    live.update(self._generate_layout())

                if self.interactive.get_and_reset_force_save() or self.step_count % self.args.save_every == 0:
                    self._save_checkpoint(state, self.step_count, ckpt_path)
                    
        self._save_checkpoint(state, self.step_count, ckpt_path)
        print("\n--- Training Saved & Shut Down Gracefully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    parser.add_argument('--audio', type=str, default=None, help="Path to audio file (optional, extracted automatically if not provided)")
    parser.add_argument('--basename', type=str, default="wubu_vhf", help="Checkpoint name")
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--pixels-per-step', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=2000)
    
    parser.add_argument('--d-model', type=int, default=512, help="Core model dimension capacity")
    parser.add_argument('--latent-grid-size', type=int, default=96, help="Target Hamilton Grid Size")
    
    args = parser.parse_args()
    VideoVHFTrainer(args).train()