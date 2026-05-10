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
import subprocess
import scipy.io.wavfile as wav
import argparse

# --- VHF CANVAS CONSTANTS ---
VBI_LINES = 45
VISIBLE_H = 480
CANVAS_H = VISIBLE_H + VBI_LINES
AUDIO_HBI_WIDTH = 16 
VISIBLE_W = 640
CANVAS_W = AUDIO_HBI_WIDTH + VISIBLE_W 
TOTAL_PIXELS = CANVAS_H * CANVAS_W

class QControllerState(struct.PyTreeNode):
    q_table: jax.Array
    metric_history: jax.Array
    current_lr: jax.Array
    exploration_rate: jax.Array
    step_count: jax.Array
    last_action_idx: jax.Array
    status_code: jax.Array

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
    def __call__(self, images_rgb):
        x = images_rgb
        features = 32
        current_h, current_w = x.shape[1], x.shape[2]
        context_vectors = []
        while (current_h // 2) >= self.latent_grid_size and (current_w // 2) >= self.latent_grid_size:
            x = nn.Conv(features, (4, 4), strides=(2, 2))(x)
            x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2; current_h //= 2; current_w //= 2
        if context_vectors: context_vector = jnp.concatenate(context_vectors, axis=-1)
        else: context_vector = jnp.zeros((x.shape[0], 1))
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
    def encode(self, images_rgb): return self.encoder(images_rgb)
    def decode(self, hamilton_keys, context_vector, coords): return self.decoder(hamilton_keys, context_vector, coords)

def load_model(args):
    print(f"[*] Waking up VHF Model ({args.d_model}d, {args.latent_grid_size}L)...")
    model = VHFEndToEndModel(latent_grid_size=args.latent_grid_size, d_model=args.d_model)
    with open(args.model_path, 'rb') as f: data = pickle.load(f)
    return model, data['ema_params']

def parse_time(t_str):
    if not t_str: return None
    parts = list(map(float, t_str.split(':')))
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return parts[0]*60 + parts[1]
    if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
    return 0.0

def run_pipeline(args):
    model, params = load_model(args)
    
    if not args.skip_compression:
        print(f"\n[*] STEP 1: Compressing frames to {args.wubu}...")
        # (Skipping the re-write of compression here to keep it clean, assume you're skipping compression for tests)
        raise NotImplementedError("For time-slicing tests, please use --skip-compression and load an existing .wubu")

    print(f"\n[*] Fast-loading existing archive {args.wubu}...")
    data = np.load(args.wubu)
    keys_uint8 = data['keys']
    ctx_uint8 = data['ctx']
    fps = data.get('fps', 24.0)
    audio_data = data['audio']
    sr = data.get('sr', 44100)
    
    total_frames = keys_uint8.shape[0]
    samples_per_frame = len(audio_data) / float(total_frames) if len(audio_data) > 1 else 0

    # --- TIMELINE SLICING ---
    start_sec = parse_time(args.start_time)
    end_sec = parse_time(args.end_time)
    
    start_frame = int(start_sec * fps) if start_sec is not None else 0
    end_frame = int(end_sec * fps) if end_sec is not None else total_frames
    
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    keys_slice = keys_uint8[start_frame:end_frame]
    ctx_slice = ctx_uint8[start_frame:end_frame]
    
    start_audio = int(start_frame * samples_per_frame)
    end_audio = int(end_frame * samples_per_frame)
    audio_slice = audio_data[start_audio:end_audio]
    
    render_frames = end_frame - start_frame
    print(f"[*] Rendering Timeline Slice: Frames {start_frame} to {end_frame} ({render_frames} frames total)")
    
    # --- DECODE LOOP ---
    CHUNK_SIZE = args.decode_chunk_size
    t = jnp.arange(TOTAL_PIXELS, dtype=jnp.float32)
    x_coords = 2.0 * ((t % CANVAS_W) / (CANVAS_W - 1)) - 1.0
    y_coords = 2.0 * (jnp.floor(t / CANVAS_W) / (CANVAS_H - 1)) - 1.0
    full_coords = jnp.stack([x_coords, y_coords], axis=-1)[None, ...]
    
    @jax.jit
    def decode_chunk(k, c, coords_chunk):
        return model.apply({'params': params}, k[None, ...], c[None, ...], coords_chunk, method=model.decode)

    temp_video = "temp_video_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, float(fps), (CANVAS_W, CANVAS_H))
    
    for i in tqdm(range(render_frames)):
        k_gpu = (jnp.array(keys_slice[i], dtype=jnp.float32) / 255.0) * 2.0 - 1.0
        c_gpu = (jnp.array(ctx_slice[i], dtype=jnp.float32) / 255.0) * 2.0 - 1.0

        pred_rgb_chunks = []
        for start_idx in range(0, TOTAL_PIXELS, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, TOTAL_PIXELS)
            chunk_pred = decode_chunk(k_gpu, c_gpu, full_coords[:, start_idx:end_idx, :])
            pred_rgb_chunks.append(np.array(chunk_pred))
            
        pred_canvas = np.concatenate(pred_rgb_chunks, axis=1).reshape((CANVAS_H, CANVAS_W, 3))
        frame_uint8 = np.clip((pred_canvas + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        
        if samples_per_frame > 0:
            s_idx = int(i * samples_per_frame)
            e_idx = int((i + 1) * samples_per_frame)
            chunk = audio_slice[s_idx:e_idx]
            if chunk.max() > chunk.min(): chunk_norm = ((chunk - chunk.min()) / (chunk.max() - chunk.min()) * 255).astype(np.uint8)
            else: chunk_norm = np.zeros_like(chunk, dtype=np.uint8)
            
            target_size = CANVAS_H * AUDIO_HBI_WIDTH
            if len(chunk_norm) < target_size: padded_audio = np.pad(chunk_norm, (0, target_size - len(chunk_norm)), 'constant')
            else: padded_audio = chunk_norm[:target_size]
                
            frame_uint8[:, :AUDIO_HBI_WIDTH, :] = np.stack([padded_audio.reshape((CANVAS_H, AUDIO_HBI_WIDTH))]*3, axis=-1)

        out.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
    out.release()
    
    print(f"\n[*] Muxing final audio track into {args.output}...")
    temp_wav = "temp_mux.wav"
    wav.write(temp_wav, int(sr), audio_slice)
    subprocess.run(['ffmpeg', '-i', temp_video, '-i', temp_wav, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', args.output, '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_video); os.remove(temp_wav)
    print(f"\n[🚀] SLICE RENDERED! Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wubu', type=str, default="bbb_encoded.wubu")
    parser.add_argument('--output', type=str, default="bbb_reconstructed.mp4")
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--latent-grid-size', type=int, default=128)
    parser.add_argument('--decode-chunk-size', type=int, default=16400)
    parser.add_argument('--skip-compression', action='store_true')
    
    # TIMELINE CONTROLS
    parser.add_argument('--start-time', type=str, default=None, help="Start time e.g., '02:15' or '135' seconds")
    parser.add_argument('--end-time', type=str, default=None, help="End time e.g., '02:20' or '140' seconds")
    
    args = parser.parse_args()
    run_pipeline(args)