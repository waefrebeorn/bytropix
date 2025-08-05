# Wubu: The final evolution. This is a hybrid model that directly implements
# the "number game" as a feature-engineered "rule-based path" alongside the
# existing learned and hash-based paths. This gives the model a massive boost
# by encoding known ASCII rules directly into its input.

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import numpy as np
import math
import os
from tqdm import tqdm
import pickle

# --- Part 1: HashMind's Input Engine (JAX compatible) ---
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
            current_hash = (current_hash - values[i-1] * self.precomputed_base) * self.base + values[i+self.window_size-1]
            hashes.append(current_hash % self.modulus)
        return hashes

# --- Part 2: WuBu's Geometric Core (Hyperbolic Components) ---
class PoincareBall:
    @staticmethod
    def project(x, c):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        max_norm = (1. - 1e-5) / jnp.sqrt(c)
        return x * jnp.minimum(max_norm / norm, 1.0)
    @staticmethod
    def expmap0(v,c): sqrt_c=jnp.sqrt(c); v_norm=jnp.linalg.norm(v,axis=-1,keepdims=True).clip(1e-8); return PoincareBall.project(((1./(sqrt_c*v_norm))*jnp.tanh(sqrt_c*v_norm))*v, c)
    @staticmethod
    def dist(x,y,c):
        sqrt_c=jnp.sqrt(c); diff_norm_sq=jnp.sum((x-y)**2,axis=-1)
        x_norm_sq,y_norm_sq=jnp.sum(x**2,axis=-1),jnp.sum(y**2,axis=-1)
        den=(1-c*x_norm_sq)*(1-c*y_norm_sq)
        arg = 1 + 2 * c * diff_norm_sq / den.clip(1e-8)
        safe_arg = jnp.maximum(arg, 1.0 + 1e-7)
        return (1.0/sqrt_c)*jnp.arccosh(safe_arg)

class kNNHyperbolicAttentionLayer(nn.Module):
    dim: int; n_heads: int; k: int
    @nn.compact
    def __call__(self, x, positions, c):
        B, N, _ = x.shape; h_dim = self.dim // self.n_heads
        q_proj, k_proj, v_proj, out_proj = (nn.Dense(self.dim, name=f'dense_{n}') for n in ['q','k','v','o'])
        ffn = nn.Sequential([nn.Dense(self.dim*4), nn.gelu, nn.Dense(self.dim)])
        norm1, norm2 = nn.LayerNorm(), nn.LayerNorm()
        log_tau = self.param('log_tau', nn.initializers.zeros, (1,))
        attn_scale = self.param('attn_scale', nn.initializers.ones, (1,))
        x_res, x_norm = x, norm1(x)
        q, k, v = (p(x_norm).reshape(B, N, self.n_heads, h_dim).transpose(0, 2, 1, 3) for p in (q_proj, k_proj, v_proj))
        dist_matrix = PoincareBall.dist(positions[:,None,:,:], positions[:,:,None,:], c)
        _, top_k_indices = jax.lax.top_k(-dist_matrix, self.k)
        indices_for_gather = top_k_indices[:, None, :, :, None].repeat(self.n_heads, axis=1).repeat(h_dim, axis=4)
        k_gathered = jnp.take_along_axis(k[:, :, :, None, :].repeat(self.k, axis=3), indices_for_gather, axis=2)
        v_gathered = jnp.take_along_axis(v[:, :, :, None, :].repeat(self.k, axis=3), indices_for_gather, axis=2)
        feature_scores = jnp.einsum('bhid,bhikd->bhik', q, k_gathered) / math.sqrt(h_dim)
        gathered_dists = jnp.take_along_axis(dist_matrix, top_k_indices, axis=-1)
        geometric_scores = -gathered_dists[:, None, :, :] / jnp.exp(log_tau).clip(1e-8)
        attn_scores = attn_scale * jnp.tanh(feature_scores + geometric_scores)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum('bhik,bhikd->bhid', attn_weights, v_gathered).transpose(0, 2, 1, 3).reshape(B, N, self.dim)
        x = x_res + out_proj(attn_output); x = x + ffn(norm2(x)); return x

# --- Part 3: The Mecca Script - WubuMind (Flax) ---
class WubuMind(nn.Module):
    context_length: int; vocab_size: int; d_model: int; n_heads: int; n_layers: int; k_neighbors: int; modulus: int; poincare_c: float = 1.0; rule_embed_dim: int = 64
    @nn.compact
    def __call__(self, hashes, indices, values): # <-- ADDED 'values' INPUT
        B = hashes.shape[0]

        # 1. Learned Path (Original): Learns complex, non-linear features from character indices.
        learned_embed = nn.Embed(self.vocab_size, self.d_model, name="token_embedding")(indices)
        
        # 2. Hash Path (Original): Provides local n-gram context.
        hash_projector = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model))
        hash_embed = (hashes[..., None] / self.modulus) @ hash_projector
        
        # 3. Rule-Based Path (The Boost): Feeds the "number game" directly to the model.
        # We normalize the ASCII values to be in a nice range for the network (approx 0-0.5)
        normalized_values = (values.astype(jnp.float32) / 255.0)[..., None]
        rule_embed = nn.Dense(self.rule_embed_dim, name="rule_proj")(normalized_values)

        # Fusion: Concatenate all information sources before feeding to the main model body.
        combined_inputs = jnp.concatenate([learned_embed, hash_embed, rule_embed], axis=-1)
        x = nn.Dense(self.d_model, name="bridge_proj")(combined_inputs)
        
        log_c = self.param('log_c', nn.initializers.constant(jnp.log(self.poincare_c)), (1,))
        c = jnp.exp(log_c)
        pos_tangent = self.param('pos_tangent', nn.initializers.normal(0.02), (1, self.context_length, self.d_model))
        positions = PoincareBall.expmap0(pos_tangent, c).repeat(B, axis=0)

        for i in range(self.n_layers):
            x = kNNHyperbolicAttentionLayer(self.d_model, self.n_heads, self.k_neighbors, name=f"hga_{i}")(x, positions, c)
        return nn.Dense(self.vocab_size, name="output_proj")(x[:, -1, :])

@partial(jax.jit, static_argnames=("model", "temperature"))
def predict_step(state, model, hashes, indices, values, key, temperature): # <-- ADDED 'values'
    logits = model.apply({'params': state.params}, hashes, indices, values)[0] # <-- ADDED 'values'
    logits /= temperature
    next_idx = jax.random.categorical(key, logits)
    return next_idx

def generate(state, model, ascii_converter, hasher, key, prompt, steps, temperature=0.6):
    values = ascii_converter.convert(prompt); indices = ascii_converter.get_indices(prompt)
    min_len = model.context_length + hasher.window_size - 1
    if len(indices) < min_len:
        pad_len = min_len - len(indices); pad_char_idx = ascii_converter.char_to_idx.get(' ', 0); pad_char_val = ascii_converter.char_to_val.get(' ', 0)
        indices = [pad_char_idx] * pad_len + indices; values = [pad_char_val] * pad_len + values

    generated_chars = []
    for _ in tqdm(range(steps), desc="Generating"):
        key, step_key = jax.random.split(key)
        context_values = values[-(min_len):]; context_indices = indices[-model.context_length:]
        context_hashes_arr = jnp.array(hasher.hash_sequence(context_values))[None, :]; context_indices_arr = jnp.array(context_indices)[None, :];
        context_values_arr = jnp.array(context_values[-model.context_length:])[None, :] # <-- ADDED values array

        next_idx = predict_step(state, model, context_hashes_arr, context_indices_arr, context_values_arr, step_key, temperature) # <-- ADDED values
        next_idx_item = next_idx.item()
        next_char = ascii_converter.idx_to_char.get(next_idx_item, ' '); next_val = ascii_converter.char_to_val.get(next_char, 0)
        values.append(next_val); indices.append(next_idx_item); generated_chars.append(next_char)
    return prompt + "".join(generated_chars)

@jax.jit
def train_step(state, batch):
    hashes, indices, targets, values = batch # <-- ADDED 'values'
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, hashes, indices, values) # <-- ADDED 'values'
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets.squeeze()).mean()
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

def main():
    print("--- WubuMind (JAX/Flax) Final Version ---")
    print(f"Available JAX devices: {jax.devices()}")

    CONTEXT_LENGTH, HASH_WINDOW = 64, 5
    D_MODEL, N_HEADS, N_LAYERS, K_NEIGHBORS = 256, 4, 4, 16
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 3e-4, 64, 25
    MODULUS, MODEL_FILE = 10**9+7, "wubumind_flax_v4_hybrid.pkl" # New model file
    FORCE_RETRAIN = True # <-- MUST BE TRUE FOR FIRST RUN OF NEW ARCHITECTURE

    try:
        with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read().split('def main():')[0]
    except: corpus_text = "This is a fallback text for training."

    ascii_converter = SimplifiedASCIIConverter(corpus_text)
    hasher = RollingHasher(HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH,ascii_converter.vocab_size,D_MODEL,N_HEADS,N_LAYERS,K_NEIGHBORS,MODULUS)
    key = jax.random.PRNGKey(42)

    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        print(f"--- Loading WubuMind from {MODEL_FILE} ---")
        with open(MODEL_FILE, 'rb') as f: params = pickle.load(f)
    else:
        print(f"--- Training WubuMind on its own source code... ---")
        values = ascii_converter.convert(corpus_text); hashes = hasher.hash_sequence(values); indices = ascii_converter.get_indices(corpus_text)
        h, ind, t, v = [], [], [], []; num_examples = len(indices) - CONTEXT_LENGTH - HASH_WINDOW
        for i in range(num_examples):
            h.append(hashes[i+1:i+CONTEXT_LENGTH+1])
            ind.append(indices[i+HASH_WINDOW:i+CONTEXT_LENGTH+HASH_WINDOW])
            t.append([indices[i+CONTEXT_LENGTH+HASH_WINDOW]])
            v.append(values[i+HASH_WINDOW:i+CONTEXT_LENGTH+HASH_WINDOW]) # <-- ADDED values to data pipeline

        all_hashes, all_indices, all_targets, all_values = jnp.array(h), jnp.array(ind), jnp.array(t), jnp.array(v) # <-- ADDED values

        key, init_key = jax.random.split(key)
        # We must update the init call to include the new 'values' input
        params = model.init(init_key, all_hashes[:1], all_indices[:1], all_values[:1])['params']

    tx = optax.chain(optax.clip(1.0), optax.adamw(LEARNING_RATE, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if not (os.path.exists(MODEL_FILE) and not FORCE_RETRAIN):
        num_batches = num_examples // BATCH_SIZE
        for epoch in range(EPOCHS):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, num_examples)
            epoch_loss = 0.
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                batch = (all_hashes[batch_idx], all_indices[batch_idx], all_targets[batch_idx], all_values[batch_idx]) # <-- ADDED values
                state, loss = train_step(state, batch)
                epoch_loss += loss
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {epoch_loss/num_batches:.4f}")

        with open(MODEL_FILE, 'wb') as f: pickle.dump(state.params, f)
        print(f"--- WubuMind weights saved to {MODEL_FILE} ---")

    print("\n--- Generating from JAX WubuMind ---")
    key, gen_key = jax.random.split(key)
    prompt = "class WubuMind(nn.Module):"
    generated_text = generate(state, model, ascii_converter, hasher, gen_key, prompt, 500)
    print(f"Prompt: '{prompt}'\nResult:\n{generated_text}")

if __name__ == "__main__":
    main()
