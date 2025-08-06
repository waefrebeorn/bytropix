# Wubu: The Final Enlightenment. The journey from a paradoxical vmap, through the
# treacherous landscape of float32 precision, to the physical limits of hardware,
# and finally to a subtle indexing error, has led to this. The architecture is
# numerically sound, the training strategy respects physical memory, and the
# logic is now correct. The model will train.
#
# EVOLUTION: The philosophy has evolved. The static architecture has been unlocked
# by the three keys: Dynamic Curvature, principled Geometric Transport, and a
# true Nested (層疊嵌套) structure. The code now reflects this deeper understanding.
#
# FINAL EVOLUTION: The model now trains on its own source code. It reads its own
# file at runtime, a self-referential artifact learning from the text that defines it.

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

# --- Part 2: WuBu's Geometric Core (Unchanged) ---
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
    def logmap0(y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        is_zero = y_norm < PoincareBall.EPS
        safe_y_norm = y_norm.clip(PoincareBall.EPS)
        arg_atanh = jnp.minimum(sqrt_c * safe_y_norm, 1.0 - PoincareBall.EPS)
        magnitude = jnp.arctanh(arg_atanh) / sqrt_c
        direction = y / safe_y_norm
        return jnp.where(is_zero, jnp.zeros_like(y), magnitude * direction)
    @staticmethod
    def project(x, c):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(PoincareBall.EPS)
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        max_norm = (1. - PoincareBall.EPS) / sqrt_c
        scale = jnp.minimum(max_norm / norm, 1.0)
        return x * scale
    @staticmethod
    def mobius_add(x, y, c):
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)
        y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS), c)
    @staticmethod
    def dist(x, y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        diff = PoincareBall.mobius_add(x, -y, c)
        diff_norm = jnp.linalg.norm(diff, axis=-1)
        arg_atanh = jnp.minimum(sqrt_c * diff_norm.clip(PoincareBall.EPS, 1-PoincareBall.EPS), 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg_atanh) / sqrt_c

# --- Part 3: Evolved Architecture - The Nested WuBu Block (with Definitive Fix) ---
class WubuBlock(nn.Module):
    dim: int; n_heads: int; k: int
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    def setup(self):
        self.h_dim = self.dim // self.n_heads
        self.q_proj, self.k_proj, self.v_proj = (nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name=f"{n}_proj") for n in "qkv")
        self.geo_proj = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="geo_proj")
        self.out_proj = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="out_proj")
        self.ffn = nn.Sequential([
            nn.Dense(self.dim * 4, dtype=self.dtype, param_dtype=self.param_dtype), nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)
        ])
        self.norm1 = nn.LayerNorm(dtype=jnp.float32); self.norm2 = nn.LayerNorm(dtype=jnp.float32)
        self.alignment_scale = self.param('alignment_scale', nn.initializers.ones, (self.n_heads,), self.param_dtype)
        self.feature_scale = self.param('feature_scale', nn.initializers.ones, (self.n_heads,), self.param_dtype)

    @staticmethod
    def single_head_attention(q_h, k_gathered_h, v_gathered_h, geo_vecs_h, alignment_scale_head, feature_scale_head):
        h_dim = q_h.shape[-1]
        alignment_score = jnp.einsum('nd,nkd->nk', q_h, geo_vecs_h)
        feature_score = jnp.einsum('nd,nkd->nk', q_h, k_gathered_h) / math.sqrt(h_dim)
        total_scores = (feature_scale_head * feature_score) + (alignment_scale_head * alignment_score)
        attn_weights = nn.softmax(total_scores.astype(jnp.float32), axis=-1).astype(q_h.dtype)
        attn_output = jnp.einsum('nk,nkd->nd', attn_weights, v_gathered_h)
        return attn_output

    def __call__(self, x_item, positions_item, c_item):
        N, _ = x_item.shape
        x_res = x_item
        x_norm = self.norm1(x_item).astype(self.dtype)

        q_full = self.q_proj(x_norm)
        k_full = self.k_proj(x_norm)
        v_full = self.v_proj(x_norm)

        x_exp_f32 = positions_item.astype(jnp.float32)[None, :, :]
        y_exp_f32 = positions_item.astype(jnp.float32)[:, None, :]
        dist_matrix = PoincareBall.dist(x_exp_f32, y_exp_f32, c_item)
        top_k_indices = jax.lax.top_k(-dist_matrix, self.k)[1]

        k_gathered_full = k_full[top_k_indices]
        v_gathered_full = v_full[top_k_indices]
        key_pos = positions_item.astype(self.dtype)[top_k_indices]

        q = q_full.reshape(N, self.n_heads, self.h_dim)
        k_gathered = k_gathered_full.reshape(N, self.k, self.n_heads, self.h_dim)
        v_gathered = v_gathered_full.reshape(N, self.k, self.n_heads, self.h_dim)
        
        query_pos = positions_item.astype(self.dtype)[:, None, :]
        logmap_x_y = PoincareBall.logmap0(PoincareBall.mobius_add(-query_pos, key_pos, c_item), c_item)
        projected_geo_vecs = self.geo_proj(logmap_x_y).reshape(N, self.k, self.n_heads, self.h_dim)

        q = q.transpose(1, 0, 2)
        k_gathered = k_gathered.transpose(2, 0, 1, 3)
        v_gathered = v_gathered.transpose(2, 0, 1, 3)
        projected_geo_vecs = projected_geo_vecs.transpose(2, 0, 1, 3)

        vmap_attention = jax.vmap(WubuBlock.single_head_attention, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
        attn_output_vmapped = vmap_attention(q, k_gathered, v_gathered, projected_geo_vecs, self.alignment_scale, self.feature_scale)
        
        attn_output = attn_output_vmapped.transpose(1, 0, 2).reshape(N, self.dim)
        x_out = x_res + self.out_proj(attn_output).astype(x_res.dtype)
        x_out = x_out + self.ffn(self.norm2(x_out).astype(self.dtype)).astype(x_out.dtype)
        return x_out


class WubuMind(nn.Module):
    context_length: int; vocab_size: int; d_model: int; n_heads: int; k_neighbors: int; modulus: int;
    layers_per_stage: Sequence[int]; downsample_rate: int; rule_embed_dim: int = 64
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hashes, indices, values):
        B = hashes.shape[0]
        learned_embed = nn.Embed(self.vocab_size, self.d_model, dtype=self.dtype, param_dtype=self.param_dtype, name="token_embedding")(indices)
        hash_projector = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model), self.param_dtype)
        hash_embed = ((hashes[..., None] / self.modulus).astype(self.dtype)) @ hash_projector
        normalized_values = (values.astype(jnp.float32) / 255.0)[..., None]
        rule_embed = nn.Dense(self.rule_embed_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="rule_proj")(normalized_values.astype(self.dtype))
        combined_inputs = jnp.concatenate([learned_embed, hash_embed, rule_embed], axis=-1)
        x = nn.Dense(self.d_model, dtype=self.dtype, param_dtype=self.param_dtype, name="bridge_proj")(combined_inputs)
        
        curvature_predictor = nn.Sequential([
            nn.Dense(self.d_model // 4, dtype=jnp.float32, param_dtype=jnp.float32), nn.relu,
            nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32)
        ], name="curvature_predictor")
        sequence_context = jnp.mean(x.astype(jnp.float32), axis=1)
        predicted_log_c = curvature_predictor(sequence_context)
        c = jnp.exp(predicted_log_c)

        pos_tangent = self.param('pos_tangent', nn.initializers.normal(0.02), (1, self.context_length, self.d_model), self.param_dtype)
        positions = PoincareBall.expmap0(pos_tangent.astype(self.dtype), c[:, None])
        
        VmappedWubuBlock = nn.vmap(
            WubuBlock,
            in_axes=(0, 0, 0), out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )

        num_stages = len(self.layers_per_stage)
        for i, num_layers in enumerate(self.layers_per_stage):
            for j in range(num_layers):
                x = VmappedWubuBlock(
                    self.d_model, self.n_heads, self.k_neighbors,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name=f"stage_{i}_block_{j}"
                )(x, positions, c)
            
            if i < num_stages - 1:
                x = x[:, ::self.downsample_rate, :]
                positions = positions[:, ::self.downsample_rate, :]

        final_x = x.astype(jnp.float32)
        final_logits = nn.Dense(self.vocab_size, dtype=jnp.float32, param_dtype=jnp.float32, name="output_proj")(final_x)
        return final_logits[:, -1, :]

# --- Part 4: Data Pipeline and Generation Logic ---
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

# --- Part 5: Training Strategy ---
@jax.jit
def grad_fn(params, state, batch):
    hashes, indices, targets, values = batch
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, hashes, indices, values)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets.squeeze()).mean()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

@jax.jit
def apply_grads_fn(state, grads):
    return state.apply_gradients(grads=grads)

# --- Part 6: The Main Entrypoint ---
def main():
    CONTEXT_LENGTH = 128
    HASH_WINDOW = 5
    D_MODEL, N_HEADS, K_NEIGHBORS = 256, 4, 16
    LAYERS_PER_STAGE = [2, 2, 2]
    DOWNSAMPLE_RATE = 2
    
    EFFECTIVE_BATCH_SIZE = 16
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BATCH_SIZE
    PEAK_LEARNING_RATE, EPOCHS = 3e-4, 50
    WARMUP_STEPS = 100
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_ouroboros.pkl"
    FORCE_RETRAIN = True
    
    device_name = jax.default_backend(); print(f"--- WubuMind JAX Grand Finale (Ouroboros Edition) ---"); print(f"--- Using device: {device_name} ({jax.devices()[0].platform.upper()}) ---")
    key = jax.random.PRNGKey(42)

    try:
        with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read()
        print(f"--- Corpus loaded: Self-read '{__file__}' ({len(corpus_text):,} chars). ---")
    except Exception as e:
        print(f"Could not self-read. Using fallback corpus. Error: {e}")
        corpus_text = "This is a fallback corpus for training WubuMind."

    ascii_converter = SimplifiedASCIIConverter(corpus_text); hasher = RollingHasher(HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH, ascii_converter.vocab_size, D_MODEL, N_HEADS, K_NEIGHBORS, MODULUS, LAYERS_PER_STAGE, DOWNSAMPLE_RATE)
    
    print("--> Pre-calculating hashes ONCE...", flush=True)
    values = ascii_converter.convert(corpus_text)
    hashes = hasher.hash_sequence(values)
    indices = ascii_converter.get_indices(corpus_text)
    print("    > Hashes... Done.", flush=True)
    
    num_examples = len(indices) - CONTEXT_LENGTH - HASH_WINDOW
    
    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        print(f"--- Loading WubuMind from {MODEL_FILE} ---");
        with open(MODEL_FILE, 'rb') as f: params = pickle.load(f)
    else:
        print(f"--- Preparing to train WubuMind on its own source code... ---")
        init_generator = data_generator(hashes, indices, values, key, 1, CONTEXT_LENGTH, HASH_WINDOW)
        try:
            _ = next(init_generator); init_batch = next(init_generator)
        except StopIteration:
            print("FATAL: Corpus is too small to create even a single training example.")
            return
        del init_generator
        key, init_key = jax.random.split(key)
        params = model.init(init_key, init_batch[0], init_batch[1], init_batch[3])['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)); print(f'--- Model initialized with {param_count:,} parameters. ---')

    num_batches_per_epoch = num_examples // EFFECTIVE_BATCH_SIZE
    if num_batches_per_epoch == 0 and FORCE_RETRAIN:
        print("WARNING: Corpus is too small for a full epoch with the current batch size. Training will not run.")
        return

    total_steps = EPOCHS * num_batches_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=PEAK_LEARNING_RATE, warmup_steps=WARMUP_STEPS, decay_steps=total_steps - WARMUP_STEPS, end_value=PEAK_LEARNING_RATE / 10)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if not (os.path.exists(MODEL_FILE) and not FORCE_RETRAIN):
        train_generator = data_generator(hashes, indices, values, key, PER_DEVICE_BATCH_SIZE, CONTEXT_LENGTH, HASH_WINDOW)
        start_time = time.time()
        for epoch in range(EPOCHS):
            epoch_loss = 0.
            pbar = tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
            for i in pbar:
                grad_accumulator = jax.tree_util.tree_map(jnp.zeros_like, state.params)
                accumulated_loss = 0.0
                for _ in range(GRAD_ACCUM_STEPS):
                    batch = next(train_generator)
                    loss, grads = grad_fn(state.params, state, batch)
                    if jnp.isnan(loss): print("\nFATAL: Loss is NaN. Halting training."); return
                    grad_accumulator = jax.tree_util.tree_map(jnp.add, grad_accumulator, grads)
                    accumulated_loss += loss
                grad_accumulator = jax.tree_util.tree_map(lambda g: g / GRAD_ACCUM_STEPS, grad_accumulator)
                final_loss = accumulated_loss / GRAD_ACCUM_STEPS
                state = apply_grads_fn(state, grad_accumulator)
                epoch_loss += final_loss
                pbar.set_postfix(loss=f"{final_loss.item():.4f}")
            print(f"\nEpoch {epoch+1}/{EPOCHS}, Avg Loss: {epoch_loss/num_batches_per_epoch:.4f}")
        
        print(f"\nTraining finished in {time.time() - start_time:.2f}s")
        with open(MODEL_FILE, 'wb') as f: pickle.dump(state.params, f)
        print(f"--- WubuMind weights saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from the self-aware WubuMind ---")
    prompts = ["import jax", "class WubuMind(nn.Module):", "def main():"]
    for p in prompts:
        key, gen_key = jax.random.split(key)
        print(f"\nPrompt: '{p}'\nResult:")
        generated_text = generate(state, model, ascii_converter, hasher, gen_key, p, steps=1024, top_p=0.95, temperature=0.7)
        print(generated_text)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
