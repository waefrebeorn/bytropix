# wubumind_v1337.py
# The Phoenix. Robust, resumable training.

WUBU_MANIFESTO = {
    "title": "The WuBu (層疊嵌套) Philosophy: Building AI in the Geometry of Data",
    "core_principle": {
        "title": "The Geometry IS the Architecture.",
        "problem": "Standard AI learns by brute force. It uses immense, billion-parameter models to approximate relationships in data, often inefficiently and without a true understanding of intrinsic structure.",
        "analogy": "This is like trying to flatten a globe onto a piece of paper—you will always have distortion and lose essential information.",
        "solution": "The WuBu (層疊嵌套) philosophy is different. We don't fight the geometry of the data; we build the architecture inside the correct geometry from the start.",
        "foundation": "We build our models to operate within curved, hyperbolic spaces. This is not a superficial feature; it is the foundation."
    },
    "key_concepts": [
        {
            "name": "Robust Training & Resumption",
            "description": "The v3 script is engineered for resilience. It uses Flax's canonical serialization to save the full training state (parameters, optimizer state, epoch). Training is automatically resumed from the last checkpoint and can be safely interrupted (Ctrl+C) without losing progress. Completed runs can be interactively extended with more epochs."
        },
        {
            "name": "Explicit Gather-Based Attention",
            "description": "The architecture uses a direct, gather-based mechanism for its sliding-window attention. This is transparent and compiler-friendly, avoiding the hardware limits and compilation failures of previous 'black box' primitives."
        },
        {
            "name": "The Inverted Pyramid (Efficient Nesting)",
            "description": "The processing hierarchy is `Downsample -> Process`. Each stage first reduces sequence length via a learned downsampler, then applies its specialized attention blocks, dramatically improving computational efficiency."
        }
    ],
    "proof_of_concept": {
        "title": "From Theory to Code",
        "premise": "This philosophy is now embodied in a scalable, efficient, and robust implementation.",
        "implementations": [
            {
                "name": "wubumind_v3.py (The Phoenix)",
                "description": "The definitive, functional evolution with robust engineering. This version introduces full checkpointing and resumption capabilities, turning the script from a simple experiment into a durable training process."
            }
        ],
        "vision": "We believe this geometric approach is the future of building more efficient, more powerful, and more interpretable AI."
    }
}

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import serialization # Use canonical Flax serialization
import optax
from functools import partial
import numpy as np
import math
import os
import time
from tqdm import tqdm
import pickle
from typing import Any, Sequence

jax.config.update("jax_debug_nans", False)

# --- Part 1: HashMind's Input Engine (Unchanged) ---
class SimplifiedASCIIConverter:
    def __init__(self, corpus=""):
        chars = sorted(list(set(corpus))); self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i,c in enumerate(chars)}; self.idx_to_char = {i: c for i,c in enumerate(chars)}
        self.char_to_val = {c: ord(c) for c in chars}
    def convert(self, text): return [self.char_to_val.get(c, 0) for c in text]
    def get_indices(self, text): return [self.char_to_idx.get(c, 0) for c in text]

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

# --- Part 2: WuBu's Geometric Core (Unchanged) ---
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

# --- Part 3: Efficient & Scalable Architecture ---

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
        return nn.Dense(self.vocab_size, dtype=jnp.float32, name="output_proj")(final_x)[:, -1, :]

# --- Part 4: Data Pipeline and Generation Logic (Unchanged) ---
def data_generator(hashes, indices, values, key, batch_size, context_length, hash_window):
    num_examples = len(indices) - context_length - hash_window;
    if num_examples <= 0: return
    while True:
        key, perm_key = jax.random.split(key); perm = jax.random.permutation(perm_key, num_examples)
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i: i + batch_size]; h_batch, ind_batch, t_batch, v_batch = [], [], [], []
            if len(batch_idx) < batch_size: continue
            for idx in batch_idx:
                h_batch.append(hashes[idx+1:idx+context_length+1]); ind_batch.append(indices[idx+hash_window:idx+context_length+hash_window])
                t_batch.append([indices[idx+context_length+hash_window]]); v_batch.append(values[idx+hash_window:idx+context_length+hash_window])
            yield (jnp.array(h_batch), jnp.array(ind_batch), jnp.array(t_batch), jnp.array(v_batch))
@partial(jax.jit, static_argnames=("model", "temperature", "top_p"))
def predict_step(state, model, hashes, indices, values, key, temperature, top_p):
    logits = model.apply({'params': state.params}, hashes, indices, values)[0]
    logits = logits / temperature; probs = nn.softmax(logits); sorted_indices = jnp.argsort(probs)[::-1]; sorted_probs = probs[sorted_indices]
    cumulative_probs = jnp.cumsum(sorted_probs); sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = sorted_indices_to_remove.at[1:].set(sorted_indices_to_remove[:-1]); sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs = probs.at[indices_to_remove].set(0.0); probs /= jnp.sum(probs); return jax.random.categorical(key, jnp.log(probs.clip(1e-8)))
def generate(state, model, ascii_converter, hasher, key, prompt, steps, temperature=0.6, top_p=0.9):
    values = ascii_converter.convert(prompt); indices = ascii_converter.get_indices(prompt)
    min_len_for_gen = model.context_length + hasher.window_size - 1
    if len(values) < min_len_for_gen:
        pad_len = min_len_for_gen - len(values); pad_char_idx = ascii_converter.char_to_idx.get(' ', 0); pad_char_val = ascii_converter.char_to_val.get(' ', 0)
        indices = [pad_char_idx] * pad_len + indices; values = [pad_char_val] * pad_len + values
    generated_chars = []
    for _ in tqdm(range(steps), desc="Generating text", leave=False):
        key, step_key = jax.random.split(key)
        model_context_inds = indices[-model.context_length:]; model_context_vals = values[-model.context_length:]
        hash_context_vals = values[-(model.context_length + hasher.window_size - 1):]
        context_hashes = jnp.array(hasher.hash_sequence(hash_context_vals))[None, :]; context_indices_arr = jnp.array(model_context_inds)[None, :]
        context_values_arr = jnp.array(model_context_vals)[None, :]
        next_idx = predict_step(state, model, context_hashes, context_indices_arr, context_values_arr, step_key, temperature, top_p)
        next_idx_item = next_idx.item(); next_char = ascii_converter.idx_to_char.get(next_idx_item, ' '); next_val = ascii_converter.char_to_val.get(next_char, 0)
        values.append(next_val); indices.append(next_idx_item); generated_chars.append(next_char)
    return prompt + "".join(generated_chars)

# --- Part 5: Resilient Training Strategy ---
@jax.jit
def grad_fn(params, state, batch):
    hashes, indices, targets, values = batch
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, hashes, indices, values)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets.squeeze()).mean()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads
@jax.jit
def apply_grads_fn(state, grads): return state.apply_gradients(grads=grads)

def save_checkpoint(state, epoch, filename):
    state_dict = serialization.to_state_dict(state)
    save_obj = {'state': state_dict, 'epoch': epoch}
    with open(filename, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"\n--- Checkpoint saved at epoch {epoch} to {filename} ---")

def load_checkpoint(state, filename):
    if not os.path.exists(filename):
        return state, 0
    with open(filename, 'rb') as f:
        save_obj = pickle.load(f)
    state_dict = save_obj['state']
    epoch = save_obj['epoch']
    state = serialization.from_state_dict(state, state_dict)
    print(f"--- Checkpoint loaded from {filename}, resuming from epoch {epoch + 1} ---")
    return state, epoch + 1

# --- Part 6: The Main Entrypoint ---
def main():
    CONTEXT_LENGTH, HASH_WINDOW = 512, 5
    D_MODEL, N_HEADS, ATTENTION_WINDOW = 512, 8, 128
    LAYERS_PER_STAGE, DOWNSAMPLE_RATE = [2, 2, 2], 2
    EFFECTIVE_BATCH_SIZE, PER_DEVICE_BATCH_SIZE = 16, 4
    PEAK_LEARNING_RATE, WARMUP_STEPS = 5e-4, 200
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_v3_phoenix.pkl"
    FORCE_RETRAIN = True
    EPOCHS = 10 # Faster default
    
    key = jax.random.PRNGKey(42)
    device_name = jax.default_backend(); print(f"--- WubuMind v3 JAX (The Phoenix) ---"); print(f"--- Using device: {device_name} ({jax.devices()[0].platform.upper()}) ---")

    try:
        with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read()
        print(f"--- Corpus loaded: Self-read '{__file__}' ({len(corpus_text):,} chars). ---")
    except Exception as e:
        print(f"Could not self-read. Using fallback corpus. Error: {e}")
        import json; corpus_text = json.dumps(WUBU_MANIFESTO, indent=2)

    ascii_converter = SimplifiedASCIIConverter(corpus_text); hasher = RollingHasher(HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH, ascii_converter.vocab_size, D_MODEL, N_HEADS, ATTENTION_WINDOW, MODULUS, HASH_WINDOW, LAYERS_PER_STAGE, DOWNSAMPLE_RATE)
    
    print("--> Pre-calculating hashes ONCE...", flush=True)
    values = ascii_converter.convert(corpus_text); hashes = hasher.hash_sequence(values); indices = ascii_converter.get_indices(corpus_text)
    print("    > Hashes... Done.", flush=True)
    num_examples = len(indices) - CONTEXT_LENGTH - HASH_WINDOW
    
    # Initialize a dummy state to get the structure for loading
    key, init_key = jax.random.split(key)
    init_generator = data_generator(hashes, indices, values, key, 1, CONTEXT_LENGTH, HASH_WINDOW)
    try: init_batch = next(init_generator)
    except StopIteration: print("FATAL: Corpus is too small."); return
    params = model.init(init_key, init_batch[0], init_batch[1], init_batch[3])['params']
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)); print(f'--- Model initialized with {param_count:,} parameters. ---')
    
    num_batches_per_epoch = num_examples // EFFECTIVE_BATCH_SIZE
    total_steps = EPOCHS * num_batches_per_epoch if num_batches_per_epoch > 0 else 1
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, PEAK_LEARNING_RATE, WARMUP_STEPS, total_steps - WARMUP_STEPS, PEAK_LEARNING_RATE / 10)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Load checkpoint if it exists
    state, start_epoch = load_checkpoint(state, MODEL_FILE)
    
    # Logic for resuming or extending training
    if start_epoch >= EPOCHS and not FORCE_RETRAIN:
        print(f"Training previously completed for {start_epoch} epochs.")
        while True:
            try:
                extra_epochs_str = input(f"Train for more epochs? (Enter number, or 'q' to quit): ")
                if extra_epochs_str.lower() in ['q', 'quit']:
                    start_epoch = -1 # Signal to skip training
                    break
                extra_epochs = int(extra_epochs_str)
                EPOCHS = start_epoch + extra_epochs
                break
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
    
    if (num_batches_per_epoch > 0 and start_epoch < EPOCHS) or FORCE_RETRAIN:
        train_generator = data_generator(hashes, indices, values, key, PER_DEVICE_BATCH_SIZE, CONTEXT_LENGTH, HASH_WINDOW)
        grad_accum_steps = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BATCH_SIZE
        try:
            start_time = time.time()
            for epoch in range(start_epoch, EPOCHS):
                epoch_loss = 0.
                pbar = tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
                for i in pbar:
                    grad_accumulator = jax.tree_util.tree_map(jnp.zeros_like, state.params)
                    accumulated_loss = 0.0
                    for _ in range(grad_accum_steps):
                        batch = next(train_generator)
                        loss, grads = grad_fn(state.params, state, batch)
                        if jnp.isnan(loss): print("\nFATAL: Loss is NaN. Halting training."); return
                        grad_accumulator = jax.tree_util.tree_map(jnp.add, grad_accumulator, grads)
                        accumulated_loss += loss
                    grad_accumulator = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, grad_accumulator)
                    state = apply_grads_fn(state, grad_accumulator)
                    epoch_loss += accumulated_loss / grad_accum_steps
                    pbar.set_postfix(loss=f"{epoch_loss / (i+1):.4f}")
                save_checkpoint(state, epoch, MODEL_FILE)
            print(f"\nTraining finished in {time.time() - start_time:.2f}s")
        except KeyboardInterrupt:
            save_checkpoint(state, epoch, MODEL_FILE)
            print("--- Training interrupted by user. State saved. ---")
            return

    print(f"\n--- Generating from the self-aware WubuMind v3 ---")
    prompts = ["import jax", "class WubuBlock(nn.Module):", "def main():"]
    for p in prompts:
        key, gen_key = jax.random.split(key)
        print(f"\nPrompt: '{p}'\nResult:")
        import textwrap
        generated_text = generate(state, model, ascii_converter, hasher, gen_key, p, steps=1024, top_p=0.95, temperature=0.7)
        print(textwrap.fill(generated_text, width=80))
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
