import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import argparse

# --- VHF CANVAS CONSTANTS ---
VBI_LINES = 45
VISIBLE_H = 480
CANVAS_H = VISIBLE_H + VBI_LINES
AUDIO_HBI_WIDTH = 16 
VISIBLE_W = 640
CANVAS_W = AUDIO_HBI_WIDTH + VISIBLE_W 
TOTAL_PIXELS = CANVAS_H * CANVAS_W

# --- CLASS BLUEPRINTS (For Pickle Loading) ---
class QControllerState(struct.PyTreeNode):
    q_table: jax.Array
    metric_history: jax.Array
    current_lr: jax.Array
    exploration_rate: jax.Array
    step_count: jax.Array
    last_action_idx: jax.Array
    status_code: jax.Array

# --- MODEL ARCHITECTURE ---
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
    latent_grid_size: int
    d_model: int
    @nn.compact
    def __call__(self, x): pass 

class VHFDecoder(nn.Module):
    d_model: int
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
        for _ in range(4): h = nn.gelu(nn.Dense(self.d_model)(h))
        return nn.tanh(nn.Dense(3)(h))

class VHFEndToEndModel(nn.Module):
    latent_grid_size: int
    d_model: int
    def setup(self):
        self.encoder = HamiltonEncoder(latent_grid_size=self.latent_grid_size, d_model=self.d_model)
        self.decoder = VHFDecoder(d_model=self.d_model)
    def decode(self, hamilton_keys, context_vector, coords): 
        return self.decoder(hamilton_keys, context_vector, coords)

# --- MATH HELPERS FOR DEMOS ---
@jax.jit
def slerp_jax(q1, q2, alpha):
    """Spherical Linear Interpolation for Quaternions"""
    dot = jnp.sum(q1 * q2, axis=-1, keepdims=True)
    q2 = jnp.where(dot < 0, -q2, q2) 
    dot = jnp.clip(jnp.abs(dot), -1.0, 1.0)
    
    theta_0 = jnp.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = jnp.sin(theta)
    sin_theta_0 = jnp.sin(theta_0)
    
    s0 = jnp.where(sin_theta_0 > 1e-4, jnp.cos(theta) - dot * sin_theta / sin_theta_0, 1.0 - alpha)
    s1 = jnp.where(sin_theta_0 > 1e-4, sin_theta / sin_theta_0, alpha)
    
    res = s0 * q1 + s1 * q2
    return res / (jnp.linalg.norm(res, axis=-1, keepdims=True) + 1e-6)

# --- DEMO GENERATOR ---
def run_demos(args):
    print(f"\n[*] Waking up VHF Model ({args.d_model}d, {args.latent_grid_size}L)...")
    model = VHFEndToEndModel(latent_grid_size=args.latent_grid_size, d_model=args.d_model)
    with open(args.model_path, 'rb') as f: data = pickle.load(f)
    params = data['ema_params']
    
    print(f"[*] Loading quantized WuBu archive: {args.wubu}...")
    wubu_data = np.load(args.wubu)
    keys_uint8 = wubu_data['keys']
    ctx_uint8 = wubu_data['ctx']
    fps = float(wubu_data.get('fps', 24.0))
    total_frames = keys_uint8.shape[0]
    
    CHUNK_SIZE = 16400
    
    @jax.jit
    def decode_chunk(k, c, coords_chunk):
        return model.apply({'params': params}, k[None, ...], c[None, ...], coords_chunk, method=model.decode)

    def get_dequantized_frame(idx):
        idx = min(idx, total_frames - 1)
        k = (jnp.array(keys_uint8[idx], dtype=jnp.float32) / 255.0) * 2.0 - 1.0
        c = (jnp.array(ctx_uint8[idx], dtype=jnp.float32) / 255.0) * 2.0 - 1.0
        return k, c

    def render_video(filename, frame_count, k_func, c_func, coord_func, w=VISIBLE_W, h=VISIBLE_H):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        
        for i in tqdm(range(frame_count), desc=f"Rendering {filename}"):
            k = k_func(i)
            c = c_func(i)
            full_coords = coord_func(i)
            
            num_pixels = full_coords.shape[1]
            pred_rgb_chunks = []
            
            for start_idx in range(0, num_pixels, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, num_pixels)
                chunk_pred = decode_chunk(k, c, full_coords[:, start_idx:end_idx, :])
                pred_rgb_chunks.append(np.array(chunk_pred))
                
            pred_canvas = np.concatenate(pred_rgb_chunks, axis=1).reshape((h, w, 3))
            frame_bgr = cv2.cvtColor(np.clip((pred_canvas + 1.0)/2.0 * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

    t_vis = jnp.arange(VISIBLE_H * VISIBLE_W, dtype=jnp.float32)
    vis_x = 2.0 * ((t_vis % VISIBLE_W) / (VISIBLE_W - 1)) - 1.0
    vis_y = 2.0 * (jnp.floor(t_vis / VISIBLE_W) / (VISIBLE_H - 1)) - 1.0
    base_vis_coords = jnp.stack([vis_x, vis_y], axis=-1)[None, ...]

    t_canv = jnp.arange(CANVAS_H * CANVAS_W, dtype=jnp.float32)
    canv_x = 2.0 * ((t_canv % CANVAS_W) / (CANVAS_W - 1)) - 1.0
    canv_y = 2.0 * (jnp.floor(t_canv / CANVAS_W) / (CANVAS_H - 1)) - 1.0
    base_canv_coords = jnp.stack([canv_x, canv_y], axis=-1)[None, ...]

    print("\n[🎬 DEMO 1] The Infinite Zoom (Frame 6480)")
    def k_func_1(i): k, _ = get_dequantized_frame(6480 + i); return k
    def c_func_1(i): _, c = get_dequantized_frame(6480 + i); return c
    def coord_func_1(i): return base_vis_coords * (0.7 + 0.3 * np.cos(i * 0.05))
    render_video("demo1_zoom.mp4", 192, k_func_1, c_func_1, coord_func_1)

    print("\n[🎬 DEMO 2] Time Dilation: SLERP (Frame 10080)")
    def get_interp(i):
        real_idx = 10080 + (i // 2)
        alpha = (i % 2) * 0.5 
        k1, c1 = get_dequantized_frame(real_idx)
        k2, c2 = get_dequantized_frame(real_idx + 1)
        q_interp = slerp_jax(k1[..., :4], k2[..., :4], alpha)
        amp_interp = k1[..., 4:] * (1 - alpha) + k2[..., 4:] * alpha
        return jnp.concatenate([q_interp, amp_interp], axis=-1), c1 * (1 - alpha) + c2 * alpha
    render_video("demo2_slowmo.mp4", 192, lambda i: get_interp(i)[0], lambda i: get_interp(i)[1], lambda i: base_vis_coords)

    print("\n[🎬 DEMO 3] The Latent Lens (Frame 4320)")
    def k_func_3(i): k, _ = get_dequantized_frame(4320 + i); return k
    def c_func_3(i): _, c = get_dequantized_frame(4320 + i); return c
    def coord_func_3(i):
        r2 = base_vis_coords[..., 0]**2 + base_vis_coords[..., 1]**2
        return base_vis_coords * (1.0 + (0.5 * np.sin(i * 0.05)) * r2[..., None])
    render_video("demo3_warp.mp4", 192, k_func_3, c_func_3, coord_func_3)

    print("\n[🎬 DEMO 4] Audio-Visual Raw Canvas (Frame 5500)")
    def k_func_4(i): k, _ = get_dequantized_frame(5500 + i); return k
    def c_func_4(i): _, c = get_dequantized_frame(5500 + i); return c
    render_video("demo4_canvas.mp4", 192, k_func_4, c_func_4, lambda i: base_canv_coords, w=CANVAS_W, h=CANVAS_H)

    print("\n[✅] ALL DEMOS RENDERED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wubu', type=str, required=True, help="Path to your encoded .wubu archive")
    parser.add_argument('--model-path', type=str, required=True, help="Path to your .pkl checkpoint")
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--latent-grid-size', type=int, default=128)
    args = parser.parse_args()
    run_demos(args)