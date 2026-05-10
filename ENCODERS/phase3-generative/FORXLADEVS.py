# Wubu: The Philosophy Made Flesh. The architecture is now built in the geometry of the data.
# The `kNNHyperbolicAttentionLayer` has been replaced with the `GeometricallyAlignedAttentionLayer`.
# This new engine computes attention based on the alignment of content vectors with the
# geometric paths (geodesics) in the hyperbolic space.
#
# Refactored the attention score calculation to use matmul instead of einsum
# a deep XLA compiler bug triggered by the complexity 

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import numpy as np
import math
import os
import requests
import time
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
            current_hash = ((current_hash - values[i-1] * self.precomputed_base) * self.base + values[i+self.window_size-1]) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes.append(current_hash)
        return hashes

# --- Part 2: WuBu's Geometric Core (Hyperbolic Components) ---
class PoincareBall:
    @staticmethod
    def project(x, c):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        max_norm = (1. - 1e-5) / jnp.sqrt(c); return x * jnp.minimum(max_norm / norm, 1.0)
    @staticmethod
    def expmap0(v,c): sqrt_c=jnp.sqrt(c); v_norm=jnp.linalg.norm(v,axis=-1,keepdims=True).clip(1e-8); return PoincareBall.project(((1./(sqrt_c*v_norm))*jnp.tanh(sqrt_c*v_norm))*v, c)
    @staticmethod
    def dist(x,y,c):
        sqrt_c=jnp.sqrt(c); diff_norm_sq=jnp.sum((x-y)**2,axis=-1)
        x_norm_sq,y_norm_sq=jnp.sum(x**2,axis=-1),jnp.sum(y**2,axis=-1)
        den=(1-c*x_norm_sq)*(1-c*y_norm_sq); arg = 1 + 2 * c * diff_norm_sq / den.clip(1e-8)
        safe_arg = jnp.maximum(arg, 1.0 + 1e-7); return (1.0/sqrt_c)*jnp.arccosh(safe_arg)
    @staticmethod
    def logmap0(y, c):
        sqrt_c = jnp.sqrt(c)
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True).clip(1e-8)
        # To prevent arctanh(1) -> inf, we clip the argument
        arg = jnp.minimum(sqrt_c * y_norm, 1.0 - 1e-7)
        return (1. / (sqrt_c * y_norm)) * jnp.arctanh(arg) * y
    @staticmethod
    def mobius_add(x, y, c):
        x2 = jnp.sum(x * x, axis=-1, keepdims=True); y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return num / den.clip(1e-8)

# --- The Embodiment of the WuBu Philosophy ---
class GeometricallyAlignedAttentionLayer(nn.Module):
    dim: int; n_heads: int; k: int
    @nn.compact
    def __call__(self, x, positions, c):
        B, N, _ = x.shape
        q_proj, k_proj, v_proj, out_proj = (nn.Dense(self.dim) for _ in ['q','k','v','o'])
        ffn = nn.Sequential([nn.Dense(self.dim*4), nn.gelu, nn.Dense(self.dim)])
        norm1, norm2 = nn.LayerNorm(), nn.LayerNorm()
        alignment_scale = self.param('alignment_scale', nn.initializers.ones, (1,))
        feature_scale = self.param('feature_scale', nn.initializers.ones, (1,))
        
        x_res, x_norm = x, norm1(x)
        q = q_proj(x_norm); k = k_proj(x_norm); v = v_proj(x_norm)
        
        dist_matrix = PoincareBall.dist(positions[:,None,:,:], positions[:,:,None,:], c)
        _, top_k_indices = jax.lax.top_k(-dist_matrix, self.k)
        
        query_pos = positions[:, :, None, :].repeat(self.k, axis=2)
        key_pos_indices = top_k_indices[..., None].repeat(self.dim, axis=-1)
        key_pos = jnp.take_along_axis(positions[:, None, :, :].repeat(N, axis=1), key_pos_indices, axis=2)
        
        logmap_x_y = PoincareBall.logmap0(PoincareBall.mobius_add(-query_pos, key_pos, c), c)
        geo_direction_vectors = nn.LayerNorm(name="geo_vec_norm")(logmap_x_y)
        
        k_content_indices = top_k_indices[..., None].repeat(self.dim, axis=-1)
        gathered_k = jnp.take_along_axis(k[:, None, :, :].repeat(N, axis=1), k_content_indices, axis=2)
        gathered_v = jnp.take_along_axis(v[:, None, :, :].repeat(N, axis=1), k_content_indices, axis=2)
        
        q_reshaped = q[:, :, None, :].repeat(self.k, axis=2)
        
        # --- XLA COMPILER FIX: Use matmul for clarity and compiler stability ---
        alignment_score = jnp.matmul(q_reshaped[..., None, :], geo_direction_vectors[..., :, None]).squeeze((-1, -2))
        feature_score = jnp.matmul(q_reshaped[..., None, :], gathered_k[..., :, None]).squeeze((-1, -2))
        
        total_scores = feature_scale * feature_score + alignment_scale * alignment_score
        attn_weights = nn.softmax(total_scores, axis=-1)
        
        attn_output = jnp.einsum('bnk,bnkd->bnd', attn_weights, gathered_v)
        
        x = x_res + out_proj(attn_output); x = x + ffn(norm2(x)); return x

# --- Part 3: The Mecca Script - WubuMind (Flax) ---
class WubuMind(nn.Module):
    context_length: int; vocab_size: int; d_model: int; n_heads: int; n_layers: int; k_neighbors: int; modulus: int; poincare_c: float = 1.0; rule_embed_dim: int = 64
    @nn.compact
    def __call__(self, hashes, indices, values):
        B = hashes.shape[0]; learned_embed = nn.Embed(self.vocab_size, self.d_model, name="token_embedding")(indices)
        hash_projector = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model))
        hash_embed = (hashes[..., None] / self.modulus) @ hash_projector
        normalized_values = (values.astype(jnp.float32) / 255.0)[..., None]
        rule_embed = nn.Dense(self.rule_embed_dim, name="rule_proj")(normalized_values)
        combined_inputs = jnp.concatenate([learned_embed, hash_embed, rule_embed], axis=-1)
        x = nn.Dense(self.d_model, name="bridge_proj")(combined_inputs)
        log_c = self.param('log_c', nn.initializers.constant(jnp.log(self.poincare_c)), (1,)); c = jnp.exp(log_c)
        pos_tangent = self.param('pos_tangent', nn.initializers.normal(0.02), (1, self.context_length, self.d_model))
        positions = PoincareBall.expmap0(pos_tangent, c).repeat(B, axis=0)
        for i in range(self.n_layers): x = GeometricallyAlignedAttentionLayer(self.d_model, self.n_heads, self.k_neighbors, name=f"hga_{i}")(x, positions, c)
        return nn.Dense(self.vocab_size, name="output_proj")(x[:, -1, :])

def data_generator(corpus_text, ascii_converter, hasher, key, batch_size, context_length, hash_window):
    values = ascii_converter.convert(corpus_text)
    print("--> Pre-calculating hashes for the entire corpus (one-time cost)...")
    hashes = hasher.hash_sequence(values); print("    > Hashes... Done.")
    indices = ascii_converter.get_indices(corpus_text)
    num_examples = len(indices) - context_length - hash_window
    while True:
        key, perm_key = jax.random.split(key); perm = jax.random.permutation(perm_key, num_examples)
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i : i + batch_size]; h_batch, ind_batch, t_batch, v_batch = [], [], [], []
            if len(batch_idx) < batch_size: continue # Skip partial batches
            for idx in batch_idx:
                h_batch.append(hashes[idx+1 : idx+context_length+1])
                ind_batch.append(indices[idx+hash_window : idx+context_length+hash_window])
                t_batch.append([indices[idx+context_length+hash_window]])
                v_batch.append(values[idx+hash_window : idx+context_length+hash_window])
            yield (jnp.array(h_batch), jnp.array(ind_batch), jnp.array(t_batch), jnp.array(v_batch))

@partial(jax.jit, static_argnames=("model", "temperature", "top_p"))
def predict_step(state, model, hashes, indices, values, key, temperature, top_p):
    logits = model.apply({'params': state.params}, hashes, indices, values)[0]
    logits = logits / temperature; probs = nn.softmax(logits)
    sorted_indices = jnp.argsort(probs)[::-1]; sorted_probs = probs[sorted_indices]
    cumulative_probs = jnp.cumsum(sorted_probs); sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = sorted_indices_to_remove.at[1:].set(sorted_indices_to_remove[:-1])
    sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs = probs.at[indices_to_remove].set(0.0); probs /= jnp.sum(probs)
    return jax.random.categorical(key, jnp.log(probs.clip(1e-8)))

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
        context_hashes = jnp.array(hasher.hash_sequence(hash_context_vals))[None, :]
        context_indices_arr = jnp.array(model_context_inds)[None, :]; context_values_arr = jnp.array(model_context_vals)[None, :]
        next_idx = predict_step(state, model, context_hashes, context_indices_arr, context_values_arr, step_key, temperature, top_p)
        next_idx_item = next_idx.item()
        next_char = ascii_converter.idx_to_char.get(next_idx_item, ' '); next_val = ascii_converter.char_to_val.get(next_char, 0)
        values.append(next_val); indices.append(next_idx_item); generated_chars.append(next_char)
    return prompt + "".join(generated_chars)

@jax.jit
def train_step(state, batch):
    hashes, indices, targets, values = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, hashes, indices, values)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets.squeeze()).mean()
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

def main():
    CONTEXT_LENGTH, HASH_WINDOW = 128, 5; D_MODEL, N_HEADS, N_LAYERS, K_NEIGHBORS = 256, 4, 4, 16
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-3, 64, 10
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_final_philosophy.pkl"
    DATA_URL, DATA_FILE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "tinyshakespeare.txt"
    FORCE_RETRAIN = True
    
    device_name = jax.default_backend(); print(f"--- WubuMind JAX Grand Finale ---"); print(f"--- Using device: {device_name} ({jax.devices()[0].platform.upper()}) ---")
    key = jax.random.PRNGKey(42)

    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}..."); r=requests.get(DATA_URL); r.raise_for_status()
        with open(DATA_FILE,'w',encoding='utf-8') as f: f.write(r.text); print("Download complete.")
    with open(DATA_FILE,'r',encoding='utf-8') as f: corpus_text=f.read()

    ascii_converter = SimplifiedASCIIConverter(corpus_text)
    hasher = RollingHasher(HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH,ascii_converter.vocab_size,D_MODEL,N_HEADS,N_LAYERS,K_NEIGHBORS,MODULUS)
    
    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        print(f"--- Loading WubuMind from {MODEL_FILE} ---");
        with open(MODEL_FILE, 'rb') as f: params = pickle.load(f)
    else:
        print(f"--- Preparing to train WubuMind on '{DATA_FILE}' ({len(corpus_text):,} chars)... ---")
        init_generator = data_generator(corpus_text, ascii_converter, hasher, key, 1, CONTEXT_LENGTH, HASH_WINDOW)
        init_batch = next(init_generator); del init_generator
        key, init_key = jax.random.split(key)
        params = model.init(init_key, init_batch[0], init_batch[1], init_batch[3])['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)); print(f'--- Model initialized with {param_count:,} parameters. ---')

    tx = optax.chain(optax.clip(1.0), optax.adamw(LEARNING_RATE, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if not (os.path.exists(MODEL_FILE) and not FORCE_RETRAIN):
        num_examples = len(corpus_text) - CONTEXT_LENGTH - HASH_WINDOW; num_batches = num_examples // BATCH_SIZE
        train_generator = data_generator(corpus_text, ascii_converter, hasher, key, BATCH_SIZE, CONTEXT_LENGTH, HASH_WINDOW)
        start_time = time.time()
        for epoch in range(EPOCHS):
            epoch_loss = 0.; pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
            for i in pbar:
                batch = next(train_generator)
                state, loss = train_step(state, batch)
                epoch_loss += loss; pbar.set_postfix(loss=f"{loss.item():.4f}")
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {epoch_loss/num_batches:.4f}")
        
        print(f"\nTraining finished in {time.time() - start_time:.2f}s")
        with open(MODEL_FILE, 'wb') as f: pickle.dump(state.params, f)
        print(f"--- WubuMind weights saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from the Grand Finale: WubuMind ---")
    prompts = ["Shall I compare thee to a summer's day?", "To be, or not to be, that is the question:"]
    for p in prompts:
        key, gen_key = jax.random.split(key)
        print(f"\nPrompt: '{p}'\nResult:")
        generated_text = generate(state, model, ascii_converter, hasher, gen_key, p, steps=1200)
        print(generated_text)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
