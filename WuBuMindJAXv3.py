# wubumind_v3.py
# The WuBu Sphere Alignment. Finalized.

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
            "name": "Natural Hierarchies in Hyperbolic Space",
            "description": "Hierarchical data (like the components of an image, the grammar of a sentence, or the evolution of a system) fits naturally into hyperbolic space, just as a tree fits in 3D space. The geometry itself provides a powerful, built-in inductive bias for learning these relationships efficiently."
        },
        {
            "name": "Hyperbolic Attention (WuBu Sphere Alignment)",
            "description": "This is the core of the v3 architecture. We replace the standard Euclidean dot-product attention with a mechanism based on pure hyperbolic distance. Queries and Keys are projected into the Poincare Ball. The attention score between them is a function of their geodesic distance. This means the model's 'focus' is determined by proximity within the learned, curved data manifold, a more principled approach than flat-space vector alignment."
        },
        {
            "name": "Adaptive, Nested Scales (層疊嵌套)",
            "description": "We use a 'Nesting' (層疊嵌套) approach, like a set of Russian dolls. Each level is its own adaptive hyperbolic space, with learnable curvature and scale, specializing in one level of abstraction. This is achieved via learned downsampling operators, allowing the model to process data from the finest details to the broadest context in a principled, multi-scale fashion."
        }
    ],
    "engine_room": {
        "title": "A System That Tunes Itself",
        "premise": "A complex architecture needs a sophisticated control system. We build our own optimizers and meta-controllers to guide the training process.",
        "example": {
            "name": "HAKMEMQController",
            "description": "A Q-learning agent that acts as a form of **'adaptive strain engineering,'** dynamically tuning the model's learning rate and momentum in real-time based on a rich stream of diagnostic data from the training process."
        },
        "motto": "It is a system that **learns how to learn better.**"
    },
    "proof_of_concept": {
        "title": "From Theory to Code",
        "premise": "This philosophy is not just a theory. It is implemented, working code that validates these core principles.",
        "implementations": [
            {
                "name": "wubumind_v3.py (HGA-UNet)",
                "description": "The definitive evolution. A subtle 5D broadcasting error in the hyperbolic distance calculation is resolved, stabilizing the pure geometric attention mechanism. The architecture now fully 'thinks' in curved space, fulfilling the core principle that the Geometry IS the Architecture."
            }
        ],
        "vision": "We believe this geometric approach is the future of building more efficient, more powerful, and more interpretable AI."
    }
}


import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
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

# --- Part 2: WuBu's Geometric Core (Corrected) ---
class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def expmap0(v, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        is_zero = v_norm < PoincareBall.EPS
        safe_v_norm = v_norm.clip(PoincareBall.EPS)
        magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c
        direction = v / safe_v_norm
        return jnp.where(is_zero, jnp.zeros_like(v), magnitude * direction)

    @staticmethod
    def dist(x, y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)
        y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        # Mobius addition of x and -y
        num = (1 - 2 * c * xy + c * y2) * x + (1 - c * x2) * (-y)
        den = 1 - 2 * c * xy + c * c * x2 * y2
        diff = num / den.clip(PoincareBall.EPS)
        # Distance calculation
        diff_norm = jnp.linalg.norm(diff, axis=-1)
        # FIX: Squeeze the curvature tensor before multiplying with the 4D diff_norm
        # to prevent broadcasting the result back to 5D.
        sqrt_c_4d = sqrt_c.squeeze(axis=-1)
        arg_atanh = jnp.minimum(sqrt_c_4d * diff_norm.clip(PoincareBall.EPS), 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg_atanh) / sqrt_c_4d

# --- Part 3: Evolved Architecture - The WuBu Sphere Alignment ---

class LearnedDownsampler(nn.Module):
    dim: int; rate: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, (self.rate,), (self.rate,), 'VALID', dtype=self.dtype, param_dtype=self.param_dtype, name="downsampler_conv")(x)

class WubuBlock(nn.Module):
    dim: int; n_heads: int; attention_window: int
    dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32

    def setup(self):
        self.h_dim = self.dim // self.n_heads
        self.q_proj, self.k_proj, self.v_proj = (nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name=f"{n}_proj") for n in "qkv")
        self.out_proj = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="out_proj")
        self.ffn = nn.Sequential([
            nn.Dense(self.dim * 4, dtype=self.dtype, param_dtype=self.param_dtype), nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)
        ])
        self.norm1 = nn.LayerNorm(dtype=jnp.float32); self.norm2 = nn.LayerNorm(dtype=jnp.float32)
        self.geo_scale = self.param('geo_scale', nn.initializers.ones, (self.n_heads, 1, 1), self.param_dtype)

    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        original_dtype = x.dtype; x_f32 = x.astype(jnp.float32)
        x_r, x_i = jnp.split(x_f32, 2, axis=-1)
        x_c = jax.lax.complex(x_r, x_i)
        freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
        x_rotated = x_c * freqs_cis
        return jnp.concatenate([x_rotated.real, x_rotated.imag], axis=-1).astype(original_dtype)

    def __call__(self, x, freqs_cis, c):
        B, N, _ = x.shape
        x_res = x; x_norm = self.norm1(x).astype(self.dtype)

        q = self.q_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)
        k = self.k_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)
        v = self.v_proj(x_norm).reshape(B, N, self.n_heads, self.h_dim)

        q_rot = self.apply_rotary_emb(q, freqs_cis)
        k_rot = self.apply_rotary_emb(k, freqs_cis)
        
        c_bcast = c.reshape(B, 1, 1, 1)

        q_hyper = PoincareBall.expmap0(q_rot, c_bcast)
        k_hyper = PoincareBall.expmap0(k_rot, c_bcast)

        q_hyper, k_hyper, v = [t.transpose(0, 2, 1, 3) for t in (q_hyper, k_hyper, v)]
        
        q_hyper_bcast = q_hyper[:, :, :, None, :]
        k_hyper_bcast = k_hyper[:, :, None, :, :]
        c_dist_bcast = c.reshape(B, 1, 1, 1, 1)
        
        attn_dist = PoincareBall.dist(q_hyper_bcast, k_hyper_bcast, c_dist_bcast)
        attn_scores = -self.geo_scale * attn_dist

        indices = jnp.arange(N)
        mask = (indices[:, None] >= indices[None, :]) & (indices[:, None] - indices[None, :] < self.attention_window)
        attn_scores = jnp.where(mask[None, None, :, :], attn_scores, -jnp.inf)
        attn_weights = nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        attn_output = jnp.einsum('bhnm,bhmd->bhnd', attn_weights, v)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N, self.dim)
        x_out = x_res + self.out_proj(attn_output).astype(x_res.dtype)
        x_out = x_out + self.ffn(self.norm2(x_out).astype(self.dtype)).astype(x_out.dtype)
        return x_out

class WubuMind(nn.Module):
    context_length: int; vocab_size: int; d_model: int; n_heads: int; attention_window: int; modulus: int
    layers_per_stage: Sequence[int]; downsample_rate: int; rule_embed_dim: int = 64
    max_len: int = 2048; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32

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
        
        freqs_cis = self.precompute_freqs_cis(h_dim, self.max_len)[:N]

        for i, num_layers in enumerate(self.layers_per_stage):
            stage_context = jnp.mean(x.astype(jnp.float32), axis=1)
            c = nn.softplus(nn.Dense(1, name=f"c_pred_{i}")(stage_context))
            for j in range(num_layers):
                x = WubuBlock(self.d_model, self.n_heads, self.attention_window, name=f"stage_{i}_block_{j}")(x, freqs_cis, c)
            if i < len(self.layers_per_stage) - 1:
                x = LearnedDownsampler(self.d_model, self.downsample_rate, name=f"downsampler_{i}")(x)
                freqs_cis = freqs_cis[::self.downsample_rate, :]

        final_x = x.astype(jnp.float32)
        return nn.Dense(self.vocab_size, dtype=jnp.float32, name="output_proj")(final_x)[:, -1, :]

# --- Part 4: Data Pipeline and Generation Logic (Largely Unchanged) ---
def data_generator(hashes, indices, values, key, batch_size, context_length, hash_window):
    num_examples = len(indices) - context_length - hash_window
    if num_examples <= 0: return
    while True:
        key, perm_key = jax.random.split(key); perm = jax.random.permutation(perm_key, num_examples)
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i : i + batch_size]; h_batch, ind_batch, t_batch, v_batch = [], [], [], []
            if len(batch_idx) < batch_size: continue
            for idx in batch_idx:
                h_batch.append(hashes[idx+1 : idx+context_length+1]); ind_batch.append(indices[idx+hash_window : idx+context_length+hash_window])
                t_batch.append([indices[idx+context_length+hash_window]]); v_batch.append(values[idx+hash_window : idx+context_length+hash_window])
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
    min_len_hash = model.context_length + hasher.window_size - 1
    if len(indices) < min_len_hash:
        pad_len = min_len_hash - len(indices); pad_char_idx = ascii_converter.char_to_idx.get(' ', 0); pad_char_val = ascii_converter.char_to_val.get(' ', 0)
        indices = [pad_char_idx] * pad_len + indices; values = [pad_char_val] * pad_len + values
    generated_chars = []
    for _ in tqdm(range(steps), desc="Generating text", leave=False):
        key, step_key = jax.random.split(key)
        hash_context_vals = values[-min_len_hash:]; model_context_inds = indices[-model.context_length:]; model_context_vals = values[-model.context_length:]
        context_hashes = jnp.array(hasher.hash_sequence(hash_context_vals))[None, :]; context_indices_arr = jnp.array(model_context_inds)[None, :]; context_values_arr = jnp.array(model_context_vals)[None, :]
        next_idx = predict_step(state, model, context_hashes, context_indices_arr, context_values_arr, step_key, temperature, top_p)
        next_idx_item = next_idx.item(); next_char = ascii_converter.idx_to_char.get(next_idx_item, ' '); next_val = ascii_converter.char_to_val.get(next_char, 0)
        values.append(next_val); indices.append(next_idx_item); generated_chars.append(next_char)
    return prompt + "".join(generated_chars)

# --- Part 5: Training Strategy (Unchanged) ---
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

# --- Part 6: The Main Entrypoint ---
def main():
    CONTEXT_LENGTH, HASH_WINDOW = 256, 5
    D_MODEL, N_HEADS, ATTENTION_WINDOW = 256, 4, 64
    LAYERS_PER_STAGE, DOWNSAMPLE_RATE = [2, 2, 2], 2
    EFFECTIVE_BATCH_SIZE, PER_DEVICE_BATCH_SIZE = 16, 8
    PEAK_LEARNING_RATE, EPOCHS, WARMUP_STEPS = 3e-4, 50, 100
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_v3_ouroboros.pkl"
    FORCE_RETRAIN = True
    
    key = jax.random.PRNGKey(42)
    device_name = jax.default_backend(); print(f"--- WubuMind v3 JAX (Sphere Alignment) ---"); print(f"--- Using device: {device_name} ({jax.devices()[0].platform.upper()}) ---")

    try:
        with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read()
        print(f"--- Corpus loaded: Self-read '{__file__}' ({len(corpus_text):,} chars). ---")
    except Exception as e:
        print(f"Could not self-read. Using fallback corpus. Error: {e}")
        import json; corpus_text = json.dumps(WUBU_MANIFESTO, indent=2)

    ascii_converter = SimplifiedASCIIConverter(corpus_text); hasher = RollingHasher(HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH, ascii_converter.vocab_size, D_MODEL, N_HEADS, ATTENTION_WINDOW, MODULUS, LAYERS_PER_STAGE, DOWNSAMPLE_RATE)
    
    print("--> Pre-calculating hashes ONCE...", flush=True)
    values = ascii_converter.convert(corpus_text); hashes = hasher.hash_sequence(values); indices = ascii_converter.get_indices(corpus_text)
    print("    > Hashes... Done.", flush=True)
    
    num_examples = len(indices) - CONTEXT_LENGTH - HASH_WINDOW
    
    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        print(f"--- Loading WubuMind v3 from {MODEL_FILE} ---")
        with open(MODEL_FILE, 'rb') as f: params = pickle.load(f)
    else:
        print(f"--- Preparing to train WubuMind v3 on its own source code... ---")
        init_generator = data_generator(hashes, indices, values, key, 1, CONTEXT_LENGTH, HASH_WINDOW)
        try:
            init_batch = next(init_generator)
        except StopIteration: print("FATAL: Corpus is too small."); return
        key, init_key = jax.random.split(key)
        params = model.init(init_key, init_batch[0], init_batch[1], init_batch[3])['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)); print(f'--- Model initialized with {param_count:,} parameters. ---')

    num_batches_per_epoch = num_examples // EFFECTIVE_BATCH_SIZE
    if num_batches_per_epoch == 0 and FORCE_RETRAIN: print("WARNING: Corpus too small for a full epoch."); return

    total_steps = EPOCHS * num_batches_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, PEAK_LEARNING_RATE, WARMUP_STEPS, total_steps - WARMUP_STEPS, PEAK_LEARNING_RATE / 10)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    grad_accum_steps = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BATCH_SIZE

    if not (os.path.exists(MODEL_FILE) and not FORCE_RETRAIN):
        train_generator = data_generator(hashes, indices, values, key, PER_DEVICE_BATCH_SIZE, CONTEXT_LENGTH, HASH_WINDOW)
        start_time = time.time()
        for epoch in range(EPOCHS):
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
            print(f"\nEpoch {epoch+1}/{EPOCHS}, Avg Loss: {epoch_loss/num_batches_per_epoch:.4f}")
        
        print(f"\nTraining finished in {time.time() - start_time:.2f}s")
        with open(MODEL_FILE, 'wb') as f: pickle.dump(state.params, f)
        print(f"--- WubuMind v3 weights saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from the self-aware WubuMind v3 ---")
    prompts = ["import jax", "class WubuBlock(nn.Module):", "def main():"]
    for p in prompts:
        key, gen_key = jax.random.split(key)
        print(f"\nPrompt: '{p}'\nResult:")
        generated_text = generate(state, model, ascii_converter, hasher, gen_key, p, steps=1024, top_p=0.95, temperature=0.7)
        print(textwrap.fill(generated_text, width=80))
        print("\n" + "="*80)

if __name__ == "__main__":
    import textwrap
    main()
