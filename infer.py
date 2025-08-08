# infer.py
# This script is the Oracle's Chamber, v2. It loads the trained WubuMind (The Phoenix)
# from its serialized state and provides a dynamic, interactive console for controlled generation.
# It is the performance, with the user as the conductor.

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import numpy as np
import math
import os
import time
from tqdm import tqdm
import pickle
from typing import Any, Sequence, Dict
import textwrap
import json

# Ensure we're using the same JAX configuration
jax.config.update("jax_debug_nans", False)

# --- Part 1: Core Components (Architectural consistency is paramount) ---

class StandardASCIIConverter:
    """A robust converter based on a fixed, standard ASCII vocabulary."""
    def __init__(self):
        # Using a fixed vocabulary of 97 printable ASCII chars + newline + tab
        self.chars = ['\n', '\t'] + [chr(i) for i in range(32, 127)]
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_to_val = {c: ord(c) for c in self.chars}
        # Use '?' as the unknown character, a common practice.
        self.unknown_char_idx = self.char_to_idx.get('?', 63)
        self.unknown_char_val = self.char_to_val.get('?', 63)
    def get_indices(self, text: str) -> list[int]:
        return [self.char_to_idx.get(c, self.unknown_char_idx) for c in text]
    def convert(self, text: str) -> list[int]:
        return [self.char_to_val.get(c, self.unknown_char_val) for c in text]

# --- [PASTE THE FULL, UNCHANGED WubuMind ARCHITECTURE HERE] ---
class RollingHasher:
    def __init__(self, window_size, base=31, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus, self.precomputed_base = window_size, base, modulus, pow(base, window_size - 1, modulus)
    def hash_sequence(self, values):
        if len(values) < self.window_size: return []
        hashes, current_hash = [], 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + values[i]) % self.modulus
        hashes.append(current_hash)
        for i in range(1, len(values) - self.window_size + 1):
            current_hash = ((current_hash - values[i-1] * self.precomputed_base) * self.base + values[i+self.window_size-1]) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes.append(current_hash)
        return hashes
class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def expmap0(v, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS); v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        is_zero = v_norm < PoincareBall.EPS; safe_v_norm = v_norm.clip(PoincareBall.EPS)
        magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c; direction = v / safe_v_norm
        return jnp.where(is_zero, jnp.zeros_like(v), magnitude * direction)
    @staticmethod
    def dist(x, y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        x2 = jnp.sum(x * x, axis=-1, keepdims=True); y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        num = (1 - 2 * c * xy + c * y2) * x + (1 - c * x2) * (-y)
        den = 1 - 2 * c * xy + c * c * x2 * y2; diff = num / den.clip(PoincareBall.EPS)
        diff_norm = jnp.linalg.norm(diff, axis=-1)
        arg_atanh = jnp.minimum(sqrt_c.squeeze(-1) * diff_norm.clip(PoincareBall.EPS), 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg_atanh) / sqrt_c.squeeze(-1)
class LearnedDownsampler(nn.Module):
    dim: int; rate: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, (self.rate,), (self.rate,), 'VALID', dtype=self.dtype, param_dtype=self.param_dtype, name="downsampler_conv")(x)
def create_sliding_windows(x, window_size):
    B, H, N, D_h = x.shape
    padding = jnp.zeros((B, H, window_size - 1, D_h), dtype=x.dtype)
    x_padded = jnp.concatenate([padding, x], axis=2)
    base_indices = jnp.arange(window_size)[None, :]; offsets = jnp.arange(N)[:, None]
    indices = base_indices + offsets
    return x_padded[:, :, indices, :]
class WubuBlock(nn.Module):
    dim: int; n_heads: int; attention_window: int; hash_window: int
    dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    def setup(self):
        self.h_dim = self.dim // self.n_heads
        self.q_proj, self.k_proj, self.v_proj = (nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name=f"{n}_proj") for n in "qkv")
        self.out_proj = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="out_proj")
        self.ffn = nn.Sequential([
            nn.Dense(self.dim * 4, dtype=self.dtype, param_dtype=self.param_dtype), nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)])
        self.norm1 = nn.LayerNorm(dtype=jnp.float32); self.norm2 = nn.LayerNorm(dtype=jnp.float32)
        self.geo_scale = self.param('geo_scale', nn.initializers.ones, (self.n_heads, 1, 1), self.param_dtype)
        self.hash_offset_param = self.param('hash_offset', nn.initializers.zeros, (1, 1, self.n_heads, self.h_dim), self.param_dtype)
    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        original_dtype = x.dtype; x_f32 = x.astype(jnp.float32)
        x_r, x_i = jnp.split(x_f32, 2, axis=-1); x_c = jax.lax.complex(x_r, x_i)
        freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
        x_rotated = x_c * freqs_cis
        return jnp.concatenate([x_rotated.real, x_rotated.imag], axis=-1).astype(original_dtype)
    def __call__(self, x, freqs_cis, c):
        B, N, _ = x.shape; x_res = x; x_norm = self.norm1(x).astype(self.dtype)
        q = self.q_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)
        k = self.k_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)
        v = self.v_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)
        q += self.hash_offset_param * jnp.log(1.0 + self.hash_window)
        q_rot = self.apply_rotary_emb(q, freqs_cis); k_rot = self.apply_rotary_emb(k, freqs_cis)
        c_bcast = c.reshape(B, 1, 1, 1)
        q_hyper = PoincareBall.expmap0(q_rot, c_bcast); k_hyper = PoincareBall.expmap0(k_rot, c_bcast)
        q_hyper, k_hyper, v = [t.transpose(0, 2, 1, 3) for t in (q_hyper, k_hyper, v)]
        k_hyper_windowed = create_sliding_windows(k_hyper, self.attention_window)
        v_windowed = create_sliding_windows(v, self.attention_window)
        q_hyper_bcast = q_hyper[:, :, :, None, :]; c_dist_bcast = c.reshape(B, 1, 1, 1, 1)
        attn_dist = PoincareBall.dist(q_hyper_bcast, k_hyper_windowed, c_dist_bcast)
        attn_scores = -self.geo_scale * attn_dist
        attn_weights = nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        attn_output = jnp.einsum('bhnw,bhnwd->bhnd', attn_weights, v_windowed)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N, self.dim)
        x_out = x_res + self.out_proj(attn_output).astype(x_res.dtype)
        x_out = x_out + self.ffn(self.norm2(x_out).astype(self.dtype)).astype(x_out.dtype)
        return x_out
class WubuMind(nn.Module):
    context_length: int; vocab_size: int; d_model: int; n_heads: int; attention_window: int; modulus: int
    hash_window: int; layers_per_stage: Sequence[int]; downsample_rate: int; rule_embed_dim: int = 64
    max_len: int = 4096; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(end); freqs = jnp.outer(t, freqs)
        return jnp.exp(1j * freqs)
    @nn.compact
    def __call__(self, hashes, indices, values):
        B, N = indices.shape; h_dim = self.d_model // self.n_heads
        learned_embed = nn.Embed(self.vocab_size, self.d_model, dtype=self.dtype, param_dtype=self.param_dtype, name="token_embedding")(indices)
        hash_projector = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model), self.param_dtype)
        hash_embed = ((hashes[..., None] / self.modulus).astype(self.dtype)) @ hash_projector
        normalized_values = (values.astype(jnp.float32) / 255.0)[..., None]
        rule_embed = nn.Dense(self.rule_embed_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="rule_proj")(normalized_values.astype(self.dtype))
        combined_inputs = jnp.concatenate([learned_embed, hash_embed, rule_embed], axis=-1)
        x = nn.Dense(self.d_model, dtype=self.dtype, param_dtype=self.param_dtype, name="bridge_proj")(combined_inputs)
        freqs_cis = self.precompute_freqs_cis(h_dim, self.max_len)
        for i, num_layers in enumerate(self.layers_per_stage):
            if i > 0:
                x = LearnedDownsampler(self.d_model, self.downsample_rate, name=f"downsampler_{i-1}")(x)
            current_N = x.shape[1]; stage_freqs_cis = freqs_cis[:current_N]
            stage_context = jnp.mean(x.astype(jnp.float32), axis=1)
            c = nn.softplus(nn.Dense(1, name=f"c_pred_{i}")(stage_context))
            for j in range(num_layers):
                x = WubuBlock(self.d_model, self.n_heads, self.attention_window, self.hash_window, name=f"stage_{i}_block_{j}")(x, stage_freqs_cis, c)
        final_x = x.astype(jnp.float32)
        # FIX: Return logits for the whole sequence. The consumer will slice the last token.
        return nn.Dense(self.vocab_size, dtype=jnp.float32, name="output_proj")(final_x)
# --- [END OF PASTED ARCHITECTURE] ---

# --- Part 3: Advanced Generation Logic ---

@partial(jax.jit, static_argnames=("model", "temperature", "top_p"))
def predict_step(state, model, hashes, indices, values, key, temperature, top_p):
    # Apply the model to get logits for the full sequence
    logits_seq = model.apply({'params': state.params}, hashes, indices, values)
    # Slice the logits for only the last token for prediction
    logits = logits_seq[:, -1, :]
    
    logits = logits / jnp.maximum(temperature, 1e-6) # Ensure temperature is non-zero
    probs = nn.softmax(logits)
    
    # Nucleus Sampling (top_p)
    sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
    
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    # Find indices to remove
    indices_to_remove = cumulative_probs > top_p
    # Shift one to the right and pad, so we keep the first token that exceeds top_p
    indices_to_remove = jnp.pad(indices_to_remove, [(0, 0), (1, 0)], mode='constant')[:, :-1]
    
    # Create a mask in the original unsorted order
    removal_mask = jnp.full_like(indices_to_remove, False)
    removal_mask = removal_mask.at[jnp.arange(probs.shape[0])[:, None], sorted_indices].set(indices_to_remove)

    probs = jnp.where(removal_mask, 0.0, probs)
    probs /= jnp.sum(probs, axis=-1, keepdims=True) # Re-normalize
    
    return jax.random.categorical(key, jnp.log(probs.clip(1e-8)))

def generate_stream(state, model, ascii_converter, hasher, key, prompt, gen_params):
    """Yields generated characters one by one for a streaming response."""
    # Convert the initial prompt into the required formats
    values = ascii_converter.convert(prompt)
    indices = ascii_converter.get_indices(prompt)
    
    # Pad the prompt if it's shorter than the model's context length
    context_length = model.context_length
    if len(indices) < context_length:
        pad_len = context_length - len(indices)
        pad_char_idx = ascii_converter.char_to_idx.get(' ', 0)
        pad_char_val = ascii_converter.char_to_val.get(' ', 32)
        indices = [pad_char_idx] * pad_len + indices
        values = [pad_char_val] * pad_len + values

    for _ in range(gen_params['steps']):
        key, step_key = jax.random.split(key)
        
        # Prepare the context windows for the model input
        model_context_inds = indices[-context_length:]
        model_context_vals = values[-context_length:]
        
        # The hasher needs a slightly longer window to produce hashes for the full context
        hash_window_size = hasher.window_size
        hash_context_vals = values[-(context_length + hash_window_size - 1):]
        
        # Calculate hashes and convert all contexts to JAX arrays with a batch dimension
        context_hashes_arr = jnp.array(hasher.hash_sequence(hash_context_vals))[None, :]
        context_indices_arr = jnp.array(model_context_inds)[None, :]
        context_values_arr = jnp.array(model_context_vals)[None, :]
        
        # Predict the next token index
        next_idx = predict_step(state, model, context_hashes_arr, context_indices_arr, context_values_arr,
                                step_key, gen_params['temperature'], gen_params['top_p'])
        
        # Decode the predicted index to a character
        next_idx_item = next_idx.item()
        next_char = ascii_converter.idx_to_char.get(next_idx_item, '?')
        next_val = ascii_converter.char_to_val.get(next_char, 63)
        
        # Append the new token to our running context sequences
        values.append(next_val)
        indices.append(next_idx_item)
        
        yield next_char

# --- Part 4: The Main Interactive Console ---

def load_trained_model(model_basename: str):
    """
    Loads a model and its configuration dynamically from basename.pkl and basename.json.
    """
    model_file = f"{model_basename}.pkl"
    config_file = f"{model_basename}.json"

    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print(f"FATAL: Model files not found. Ensure '{model_file}' and '{config_file}' exist.")
        return None, None, None, None, None

    print(f"--> Loading model configuration from '{config_file}'...")
    with open(config_file, 'r') as f:
        # FIX: Load JSON with lowercase keys that match WubuMind's constructor
        model_config = json.load(f)

    print("--> Initializing with universal Standard ASCII vocabulary...")
    ascii_converter = StandardASCIIConverter()
    # FIX: Use lowercase keys from the loaded config
    hasher = RollingHasher(model_config['hash_window'], modulus=model_config['modulus'])
    
    # The vocab size is determined by the converter, not the config file
    model_config['vocab_size'] = ascii_converter.vocab_size
    model = WubuMind(**model_config)

    print("--> Initializing model structure to receive weights...")
    key = jax.random.PRNGKey(0)
    # Create dummy inputs that match the model's expected context length
    dummy_indices = jnp.zeros((1, model_config['context_length']), dtype=jnp.int32)
    dummy_hashes = jnp.zeros((1, model_config['context_length']), dtype=jnp.int32)
    dummy_values = jnp.zeros((1, model_config['context_length']), dtype=jnp.int32)
    
    # Initialize parameters to get the correct structure
    params = model.init(key, dummy_hashes, dummy_indices, dummy_values)['params']
    # Create a dummy optimizer state for inference
    tx = optax.adamw(0.0) 
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    print(f"--> Loading trained weights from '{model_file}'...")
    with open(model_file, 'rb') as f:
        save_obj = pickle.load(f)
    
    # Restore the learned parameters into our structured state object
    state = serialization.from_state_dict(state, save_obj['state'])
    loaded_epoch = save_obj['epoch']
    
    print(f"--- Model 'The Phoenix' successfully resurrected from epoch {loaded_epoch}. Ready for inference. ---")
    return state, model, ascii_converter, hasher, model_config

def print_help():
    print("\n--- Oracle's Chamber Commands ---")
    print("  /help              : Show this help message.")
    print("  /temp [float]      : Set sampling temperature (e.g., /temp 0.7). Higher is more random.")
    print("  /topp [float]      : Set nucleus sampling top_p (e.g., /topp 0.95).")
    print("  /steps [int]       : Set number of characters to generate (e.g., /steps 512).")
    print("  /config            : Display the current generation and model parameters.")
    print("  /reset             : Reset generation parameters to default.")
    print("  /clear             : Clear the screen.")
    print("  /exit, /quit       : End the session.")
    print("-" * 33)

def interactive_session(state, model, converter, hasher, model_config):
    """The main interactive console loop with dynamic parameter control."""
    key = jax.random.PRNGKey(int(time.time()))
    
    defaults = {'temperature': 0.7, 'top_p': 0.95, 'steps': 768}
    gen_params = defaults.copy()

    print("\n" + "="*80)
    print("      WubuMind v3 Interactive Session (The Oracle's Chamber)")
    print("="*80)
    print("Welcome. I am ready to generate. Type '/help' for a list of commands.")
    
    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if not prompt: continue

        if prompt.lower().startswith('/'):
            parts = prompt.lower().split()
            cmd = parts[0]
            try:
                if cmd in ['/exit', '/quit']:
                    break
                elif cmd == '/help':
                    print_help()
                elif cmd == '/clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                elif cmd == '/reset':
                    gen_params = defaults.copy()
                    print("Generation parameters reset to default.")
                elif cmd == '/temp':
                    gen_params['temperature'] = float(parts[1])
                    print(f"Temperature set to {gen_params['temperature']}.")
                elif cmd == '/topp':
                    gen_params['top_p'] = float(parts[1])
                    print(f"Top_p set to {gen_params['top_p']}.")
                elif cmd == '/steps':
                    gen_params['steps'] = int(parts[1])
                    print(f"Steps set to {gen_params['steps']}.")
                elif cmd == '/config':
                    print("\n--- Current Configuration ---")
                    print("Generation Parameters:")
                    for k, v in gen_params.items(): print(f"  {k}: {v}")
                    print("\nModel Hyperparameters:")
                    for k, v in model_config.items(): print(f"  {k}: {v}")
                    print("-" * 29)
                else:
                    print(f"Unknown command: '{cmd}'. Type /help for options.")
            except (IndexError, ValueError):
                print(f"Invalid argument for command '{cmd}'. Type /help for usage.")
            continue
            
        key, gen_key = jax.random.split(key)
        
        print("\nWubuMind:")
        try:
            for char in generate_stream(state, model, converter, hasher, gen_key, prompt, gen_params):
                print(char, end='', flush=True)
        except KeyboardInterrupt:
            # Allow user to interrupt generation without quitting the session
            print("\n[Generation interrupted by user]")
        
        print() # Final newline after response

if __name__ == "__main__":
    MODEL_BASENAME = "wubumind_v3_phoenix"
    
    state, model, converter, hasher, model_config = load_trained_model(MODEL_BASENAME)

    if state:
        interactive_session(state, model, converter, hasher, model_config)

    print("\n--- Session ended. The Oracle is silent. ---")