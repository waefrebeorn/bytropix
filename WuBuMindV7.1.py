import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import time
from tqdm import tqdm
import pickle
from typing import Any, Generator, Tuple
import sys
import argparse
from collections import deque
import faiss
import signal

# --- Environment Setup ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
jax.config.update('jax_threefry_partitionable', True)
jax.config.update('jax_default_matmul_precision', 'bfloat16')

# --- Import Corpus and Tokenizer ---
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
except ImportError: print("[FATAL] `tokenizers` not found. `pip install tokenizers`."), sys.exit(1)
try:
    import CORPUS
except ImportError: print("[FATAL] CORPUS.py not found."), sys.exit(1)

# --- Helper Functions ---
def stream_text_from_corpus_data(data: Any) -> Generator[str, None, None]:
    if isinstance(data, str): yield data
    elif isinstance(data, dict):
        for v in data.values(): yield from stream_text_from_corpus_data(v)
    elif isinstance(data, list):
        for item in data: yield from stream_text_from_corpus_data(item)

class WubuTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        if os.path.exists(tokenizer_path): self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else: self.tokenizer = None
    def train(self, corpus_iterator, vocab_size):
        print("--- Training tokenizer... ---")
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<PAD>", "<UNK>"])
        self.tokenizer.train_from_iterator(corpus_iterator, trainer)
        self.tokenizer.save(self.tokenizer_path)
        print(f"--- Tokenizer trained. Vocab: {self.get_vocab_size()}. Saved to {self.tokenizer_path} ---")
    def get_vocab_size(self): return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    def encode(self, text): return self.tokenizer.encode(text).ids if self.tokenizer else []
    def decode(self, ids):
        if not self.tokenizer: return ""
        return self.tokenizer.decode(ids)

class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def project(x):
        x_f32 = x.astype(jnp.float32)
        norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        max_norm = 1.0 - PoincareBall.EPS
        projected_f32 = jnp.where(norm_sq >= 1.0, x_f32 / jnp.sqrt(norm_sq + PoincareBall.EPS) * max_norm, x_f32)
        return projected_f32.astype(x.dtype)
    @staticmethod
    def mobius_add(x, y, c=1.0):
        x_f32, y_f32 = x.astype(jnp.float32), y.astype(jnp.float32)
        x2, y2, xy = jnp.sum(x_f32*x_f32, -1, keepdims=True), jnp.sum(y_f32*y_f32, -1, keepdims=True), jnp.sum(x_f32*y_f32, -1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x_f32 + (1 - c * x2) * y_f32
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / jnp.clip(den, PoincareBall.EPS)).astype(x.dtype)
    @staticmethod
    def logmap0(y, c=1.0):
        y_f32 = y.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True)
        safe_norm = jnp.clip(y_norm, PoincareBall.EPS, 1.0 - PoincareBall.EPS)
        arctanh_val = jnp.arctanh(safe_norm)
        result_f32 = jnp.where(safe_norm > 0, arctanh_val * y_f32 / (sqrt_c * safe_norm), jnp.zeros_like(y_f32))
        return result_f32.astype(y.dtype)
    @staticmethod
    def expmap0(v, c=1.0):
        v_f32 = v.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True)
        safe_norm = jnp.clip(v_norm, PoincareBall.EPS)
        tanh_val = jnp.tanh(sqrt_c * safe_norm)
        result_f32 = jnp.where(safe_norm > 0, PoincareBall.project(tanh_val * v_f32 / (sqrt_c * safe_norm)), jnp.zeros_like(v_f32))
        return result_f32.astype(v.dtype)
    @staticmethod
    def expmap_p(p, v, c=1.0):
        p_f32, v_f32 = p.astype(jnp.float32), v.astype(jnp.float32)
        lambda_p = 2. / jnp.clip(1 - c * jnp.sum(p_f32*p_f32, -1, keepdims=True), PoincareBall.EPS)
        expmapped_v = PoincareBall.expmap0(v_f32 * lambda_p, c)
        return PoincareBall.mobius_add(p_f32, expmapped_v, c).astype(p.dtype)

# --- SFIN Inspired Complex-Valued Components ---
class ComplexEmbedding(nn.Module):
    vocab_size: int; features: int; dtype: Any
    @nn.compact
    def __call__(self, x):
        real_embed = nn.Embed(self.vocab_size, self.features, name="real_embed", dtype=self.dtype)(x)
        imag_embed = nn.Embed(self.vocab_size, self.features, name="imag_embed", dtype=self.dtype)(x)
        return real_embed, imag_embed

class ComplexLayerNorm(nn.Module):
    dtype: Any
    @nn.compact
    def __call__(self, x_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        real, imag = x_complex
        real_norm = nn.LayerNorm(dtype=self.dtype, name="real_ln")(real)
        imag_norm = nn.LayerNorm(dtype=self.dtype, name="imag_ln")(imag)
        return real_norm, imag_norm

class WaveFunctionCollapseHead(nn.Module):
    vocab_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, h_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        h_r, h_i = h_complex
        real_logits = nn.Dense(self.vocab_size, name="real_collapse", dtype=self.dtype)(h_r)
        imag_logits = nn.Dense(self.vocab_size, name="imag_collapse", dtype=self.dtype)(h_i)
        return real_logits**2 + imag_logits**2

# --- Main Model Architecture ---
class GRUCell(nn.Module):
    """A standard, real-valued GRU cell implementation."""
    d_model_total: int
    dtype: Any

    @nn.compact
    def __call__(self, carry, x):
        # This implementation has always been correct.
        xh = jnp.concatenate([x, carry], axis=-1)
        r = nn.sigmoid(nn.Dense(self.d_model_total, name="reset_gate_d", dtype=self.dtype)(xh))
        u = nn.sigmoid(nn.Dense(self.d_model_total, name="update_gate_d", dtype=self.dtype)(xh))
        
        c_in = jnp.concatenate([x, r * carry], axis=-1)
        c = nn.tanh(nn.Dense(self.d_model_total, name="candidate_gate_d", dtype=self.dtype)(c_in))
        
        new_carry = (1 - u) * carry + u * c
        return new_carry, new_carry

class DripHead(nn.Module):
    d_model_total: int
    vocab_size: int
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.d_model_comp = self.d_model_total // 2
        self.token_embed = ComplexEmbedding(self.vocab_size, self.d_model_comp, name="token_embed", dtype=self.dtype)
        self.layer_norm = ComplexLayerNorm(dtype=self.dtype, name="layer_norm")
        self.collapse_head = WaveFunctionCollapseHead(vocab_size=self.vocab_size, name="collapse_head")

        scanner = nn.scan(
            GRUCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1
        )
        # This correctly creates a full module named 'gru' in setup
        self.gru = scanner(d_model_total=self.d_model_total, dtype=self.dtype)

    def __call__(self, token_ids):
        batch_size = token_ids.shape[0]
        initial_carry_real = jnp.zeros((batch_size, self.d_model_total), dtype=self.dtype)

        token_embeds_r, token_embeds_i = self.token_embed(token_ids)
        xs_real = jnp.concatenate([token_embeds_r, token_embeds_i], axis=-1)

        # This correctly uses the pre-built module
        final_carry_real, hidden_states_real = self.gru(initial_carry_real, xs_real)
        
        h_r, h_i = jnp.split(hidden_states_real, 2, axis=-1)
        hidden_states_complex = (h_r, h_i)
        
        normalized_hidden_states = self.layer_norm(hidden_states_complex)
        logits = self.collapse_head(normalized_hidden_states)

        final_carry_r, final_carry_i = jnp.split(final_carry_real, 2, axis=-1)
        final_carry_complex = (final_carry_r, final_carry_i)
        
        return final_carry_complex, normalized_hidden_states, logits


class FunnelCakeConstructor:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        self.d_model = config['d_model']
        self.key = jax.random.PRNGKey(42)
        self.drip_head = DripHead(d_model_total=self.d_model, vocab_size=tokenizer.get_vocab_size())
        self.train_state = None; self.drip_head_params = None
        self.formation_space = deque()
        self.H_sphere_points = np.empty((0, self.d_model), dtype=np.float32)
        self.H_sphere_metadata = []; self.hyperbolic_index = None
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if not self.should_shutdown:
            print("\n--- SIGINT received. Saving/exiting when possible. Press again to force quit. ---")
            self.should_shutdown = True
        else:
            print("\n--- Forced exit. ---")
            sys.exit(1)

    def _init_drip_head(self, training=False):
        if self.drip_head_params or self.train_state: return
        mode_str = "for Training" if training else "(Random Projection Mode)"
        print(f"--- Initializing Complex Drip Head {mode_str}... ---")
        self.key, drip_key = jax.random.split(self.key)
        dummy_tokens = jnp.zeros((2, self.config['train_chunk_size']), dtype=jnp.int32)
        params = self.drip_head.init(drip_key, dummy_tokens)['params']
        param_count = sum(x.size for x in jax.tree.leaves(params))
        if training:
            tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.config['learning_rate']))
            self.train_state = train_state.TrainState.create(apply_fn=self.drip_head.apply, params=params, tx=tx)
            print(f"--- Drip Head Initialized for Training: {param_count:,} params. ---")
        else:
            self.drip_head_params = params
            print(f"--- Drip Head Initialized: {param_count:,} params. ---")

    def train(self, token_file_path, epochs=3, batch_size=256):
        self._init_drip_head(training=True)
        chunk_size = self.config['train_chunk_size']
        space_token_id = self.tokenizer.tokenizer.token_to_id("Ġ") if self.tokenizer.tokenizer else -1
        min_unique_non_space = max(1, chunk_size // 4)
        print("--- Opening memory-mapped token file for training... ---")
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')
        n_tokens = len(tokens)
        indices = list(range(0, n_tokens - (chunk_size + 10)))

        @jax.jit
        def train_step(state, batch, margin):
            anchor, positive, negative = batch
            def loss_fn(params):
                _, anchor_h_c, _ = state.apply_fn({'params': params}, anchor)
                _, positive_h_c, _ = state.apply_fn({'params': params}, positive)
                _, negative_h_c, _ = state.apply_fn({'params': params}, negative)
                ah_r, ah_i = anchor_h_c[0][:, -1, :], anchor_h_c[1][:, -1, :]
                ph_r, ph_i = positive_h_c[0][:, -1, :], positive_h_c[1][:, -1, :]
                nh_r, nh_i = negative_h_c[0][:, -1, :], negative_h_c[1][:, -1, :]
                dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2, axis=-1)
                dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2, axis=-1)
                losses = jnp.maximum(0, dist_pos - dist_neg + margin)
                return jnp.mean(losses)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        for epoch in range(epochs):
            current_margin = 0.25 + (epoch / max(1, epochs-1)) * 0.75
            print(f"\n--- Starting Epoch {epoch + 1}/{epochs} (Margin: {current_margin:.2f}, Batch Size: {batch_size}) ---")
            np.random.shuffle(indices)
            pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1} (SFIN)", total=len(indices)//batch_size)
            for i in pbar:
                if self.should_shutdown: break
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size: continue
                anchors_list, positives_list, negatives_list = [], [], []
                for start_idx in batch_indices:
                    anchors_list.append(tokens[start_idx : start_idx + chunk_size])
                    positives_list.append(tokens[start_idx + 10 : start_idx + 10 + chunk_size])
                    tries = 0
                    while tries < 100:
                        neg_start_idx = np.random.randint(0, n_tokens - chunk_size)
                        negative_chunk = tokens[neg_start_idx : neg_start_idx + chunk_size]
                        if len(set(negative_chunk)) >= min_unique_non_space:
                            negatives_list.append(negative_chunk); break
                        tries += 1
                    else: negatives_list.append(tokens[0:chunk_size])
                if len(negatives_list) != batch_size: continue
                batch_jnp = (jnp.array(anchors_list), jnp.array(positives_list), jnp.array(negatives_list))
                self.train_state, loss = train_step(self.train_state, batch_jnp, current_margin)
                pbar.set_postfix(avg_loss=f"{loss.item():.4f}")
            if self.should_shutdown: print("\n--- Interrupt honored... ---"); break
        if not self.should_shutdown: print("\n--- Training Complete ---")
        self.drip_head_params = self.train_state.params
        self.save_trained_weights(self.config['basename'])

    def save_trained_weights(self, basename):
        params_to_save = self.drip_head_params if self.train_state is None else self.train_state.params
        if params_to_save is None: print("[WARN] No Drip Head parameters found to save."); return
        print(f"--- Saving Drip Head weights to {basename}.weights.pkl ---")
        with open(f"{basename}.weights.pkl", 'wb') as f: pickle.dump(jax.device_get(params_to_save), f)
        print("--- Weights saved. ---")

    def load_trained_weights(self, basename):
        weights_file = f"{basename}.weights.pkl"
        self._init_drip_head(training=False)
        if not os.path.exists(weights_file): print("[INFO] No pre-trained weights found."); return
        print(f"--- Loading trained Drip Head weights from {weights_file} ---")
        with open(weights_file, 'rb') as f: self.drip_head_params = pickle.load(f)
        param_count = sum(x.size for x in jax.tree.leaves(self.drip_head_params))
        print(f"--- Trained weights loaded. {param_count:,} params. ---")

    def construct(self, token_file_path):
        self.should_shutdown = False
        self.load_trained_weights(self.config['basename'])
        print(f"--- Constructing Funnel Cake from memory-mapped tokens... ---")
        batch_size = 512
        pad_token_id = self.tokenizer.tokenizer.token_to_id("<PAD>") if self.tokenizer.tokenizer else 0
        max_formation_size = 2_000_000
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')
        @jax.jit
        def drip_batch_step(params, token_batch):
            _, all_hidden_complex, _ = self.drip_head.apply({'params': params}, token_batch)
            return all_hidden_complex
        pbar = tqdm(total=len(tokens), unit='tok', unit_scale=True, desc="Constructing Filament")
        for i in range(0, len(tokens), batch_size):
            if self.should_shutdown: break
            batch_tokens = tokens[i:i+batch_size]
            original_len = len(batch_tokens)
            if original_len == 0: continue
            padded_batch = np.pad(batch_tokens, (0, batch_size - original_len), 'constant', constant_values=pad_token_id)
            batch_jax = jnp.array(padded_batch, dtype=jnp.int32)[jnp.newaxis, :]
            batch_hidden_states_complex = drip_batch_step(self.drip_head_params, batch_jax)
            hidden_states_r = np.array(batch_hidden_states_complex[0]).squeeze(0)[:original_len]
            hidden_states_i = np.array(batch_hidden_states_complex[1]).squeeze(0)[:original_len]
            hidden_states_collapsed = np.concatenate([hidden_states_r, hidden_states_i], axis=-1)
            for j in range(original_len):
                self.formation_space.append({'drip_head_state': hidden_states_collapsed[j], 'token_id': batch_tokens[j].item()})
            while len(self.formation_space) >= max_formation_size: self._solidify()
            while len(self.formation_space) >= self.config['solidify_chunk_size']: self._solidify()
            pbar.update(original_len)
            pbar.set_postfix(solids=f"{len(self.H_sphere_metadata):,}", formation_q=f"{len(self.formation_space):,}")
        if self.formation_space: self._solidify(force_all=True)
        pbar.close()
        print(f"\n--- Construction Complete. Solids: {self.H_sphere_points.shape[0]} ---")
        self.save(self.config['basename'])

    def _solidify(self, force_all=False):
        chunk_size = len(self.formation_space) if force_all else self.config['solidify_chunk_size']
        if chunk_size == 0: return
        chunk = [self.formation_space.popleft() for _ in range(chunk_size)]
        drip_states = np.array([item['drip_head_state'] for item in chunk]).astype(np.float32)
        tangent_vec = np.mean(drip_states, axis=0)
        if self.H_sphere_points.shape[0] < self.config['knn']:
            new_point = np.array(PoincareBall.expmap0(jnp.array(tangent_vec)))
        else:
            _, indices = self.hyperbolic_index.search(np.expand_dims(tangent_vec, 0), self.config['knn'])
            neighbors = self.H_sphere_points[indices.flatten()]
            avg_tangent = np.mean(jax.vmap(PoincareBall.logmap0)(jnp.array(neighbors)), axis=0)
            blended_vec = (tangent_vec * 0.5) + (np.array(avg_tangent) * 0.5)
            new_point = np.array(PoincareBall.expmap0(jnp.array(blended_vec)))
        self.H_sphere_points = np.vstack([self.H_sphere_points, new_point])
        if self.hyperbolic_index is None: self.hyperbolic_index = faiss.IndexFlatL2(self.d_model)
        self.hyperbolic_index.add(np.array(self.H_sphere_points[-1:], dtype=np.float32))
        self.H_sphere_metadata.append({'token_ids': [item['token_id'] for item in chunk]})

    def save(self, basename):
        print(f"--- Saving Funnel Cake to {basename}.cake ---")
        state = {'config': self.config, 'H_sphere_points': self.H_sphere_points, 'H_sphere_metadata': self.H_sphere_metadata}
        with open(f"{basename}.cake", 'wb') as f: pickle.dump(state, f)
        self.save_trained_weights(basename)
        print("--- Save complete. ---")

    def _precompile_generation(self):
        print("--- Pre-compiling generation function for common prompt lengths... ---")
        @jax.jit
        def get_model_outputs_jitted(params, tokens):
            # This jitted function gets all outputs from the model.
            return self.drip_head.apply({'params': params}, tokens)
        get_model_outputs_jitted(self.drip_head_params, jnp.ones((1, 1), dtype=jnp.int32))
        for length in [8, 16, 32, 64]:
            print(f"    Compiling for length {length}...")
            dummy_tokens = jnp.ones((1, length), dtype=jnp.int32)
            get_model_outputs_jitted(self.drip_head_params, dummy_tokens)
        print("--- Pre-compilation complete. ---")

    def load(self, basename):
        print(f"--- Loading Funnel Cake from {basename}.cake ---")
        with open(f"{basename}.cake", 'rb') as f: state = pickle.load(f)
        self.config = state['config']
        self.load_trained_weights(basename)
        self.H_sphere_points = state['H_sphere_points']
        self.H_sphere_metadata = state['H_sphere_metadata']
        if self.H_sphere_points.shape[0] > 0:
            self.hyperbolic_index = faiss.IndexFlatL2(self.d_model)
            self.hyperbolic_index.add(self.H_sphere_points.astype(np.float32))
        print(f"--- Funnel Cake loaded. Contains {self.H_sphere_points.shape[0]} solidified points. ---")
        self._precompile_generation()

    def generate(self, prompt, max_new=200, momentum=0.8, temp=0.8):
        self.should_shutdown = False
        if self.hyperbolic_index is None or self.H_sphere_points.shape[0] == 0:
            print("\n[ERROR] Funnel Cake is empty.")
            return
        print(f"\n\033[1;32m{prompt}\033[0m", end='', flush=True)

        @jax.jit
        def get_model_outputs(params, tokens):
            return self.drip_head.apply({'params': params}, tokens)

        prompt_tokens = self.tokenizer.encode(prompt)
        if not prompt_tokens: prompt_tokens = [self.tokenizer.tokenizer.token_to_id("<PAD>")]

        ### --- CHANGE START --- ###
        # Logic fix: We must use the *normalized* hidden state to query the hyperbolic index,
        # because the index was built using those same normalized states during construction.
        # The original code used the raw GRU state (final_carry), creating a mismatch.

        # 1. Process the prompt to get the initial NORMALIZED hidden state.
        _, normalized_h_c, _ = get_model_outputs(self.drip_head_params, jnp.array([prompt_tokens]))
        # Extract the state from the very last token in the prompt sequence
        last_h_r = normalized_h_c[0][:, -1, :]
        last_h_i = normalized_h_c[1][:, -1, :]
        hidden_state_collapsed = jnp.concatenate([last_h_r.squeeze(), last_h_i.squeeze()])
        ### --- CHANGE END --- ###

        _, start_indices = self.hyperbolic_index.search(np.array(hidden_state_collapsed, dtype=np.float32).reshape(1, -1), 1)
        current_point = jnp.array(self.H_sphere_points[start_indices[0][0]])
        velocity_vector = jnp.zeros_like(current_point)
        current_tokens = list(prompt_tokens)

        for i in range(max_new):
            if self.should_shutdown: print("\n--- Interrupt honored. ---"); break

            temp_start, temp_end = temp, max(0.1, temp * 0.5)
            current_temp = temp_start + (temp_end - temp_start) * (i / max(max_new - 1, 1))

            intent_vector = PoincareBall.logmap0(current_point) - hidden_state_collapsed
            _, neighbor_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), self.config['knn_sampling'])
            neighbors = jnp.array(self.H_sphere_points[neighbor_indices.flatten()])
            tangent_vectors = jax.vmap(PoincareBall.logmap0)(neighbors)
            local_flow_vector = jnp.mean(tangent_vectors, axis=0)

            new_velocity = (velocity_vector * momentum) + (intent_vector * 0.2) + (local_flow_vector * (1 - momentum))
            velocity_vector = new_velocity / jnp.linalg.norm(new_velocity).clip(1e-6)
            current_point = PoincareBall.expmap_p(current_point, velocity_vector * self.config['geodesic_step_size'])

            _, closest_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), 1)
            chosen_neighbor_idx = closest_indices.flatten()[0]
            retrieved_chunk_tokens = self.H_sphere_metadata[chosen_neighbor_idx]['token_ids']
            if not retrieved_chunk_tokens: continue

            # Get logits based on the *current* full sequence of tokens.
            _, _, logits = get_model_outputs(self.drip_head_params, jnp.array([current_tokens]))
            next_step_logits = logits.squeeze()[-1, :]

            allowed_tokens = jnp.array(list(set(retrieved_chunk_tokens)))
            mask = jnp.full(self.tokenizer.get_vocab_size(), -jnp.inf)
            mask = mask.at[allowed_tokens].set(0.0)
            masked_logits = next_step_logits + mask

            probs = nn.softmax(masked_logits / current_temp)
            self.key, subkey = jax.random.split(self.key)
            next_token_id = jax.random.categorical(subkey, jnp.log(probs.clip(1e-9)))

            decoded_token = self.tokenizer.decode([next_token_id.item()])
            print(decoded_token.replace('Ġ', ' '), end='', flush=True)

            current_tokens.append(next_token_id.item())

            ### --- CHANGE START --- ###
            # Update the guiding hidden state for the next step. Again, we must use the
            # normalized state from the last token of the *new, longer* sequence.
            _, normalized_h_c_new, _ = get_model_outputs(self.drip_head_params, jnp.array([current_tokens]))
            last_h_r_new = normalized_h_c_new[0][:, -1, :]
            last_h_i_new = normalized_h_c_new[1][:, -1, :]
            hidden_state_collapsed = jnp.concatenate([last_h_r_new.squeeze(), last_h_i_new.squeeze()])
            ### --- CHANGE END --- ###
        print()


def main():
    parser = argparse.ArgumentParser(description="WubuMind Funnel Cake Constructor v22.2 (Robust Compact Model)")
    parser.add_argument('command', choices=['pretokenize', 'train', 'construct', 'generate'], help="The command to execute.")
    parser.add_argument('--basename', type=str, default="wubumind_funnel_cake_v1", help="Basename for model files.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4096, help="Batch size for training. Adjust based on VRAM.")
    parser.add_argument('--temp', type=float, default=0.9, help="Initial temperature for generation.")
    parser.add_argument('--momentum', type=float, default=0.85, help="Momentum for the missile guidance system.")
    parser.add_argument('--max-new', type=int, default=200, help="Maximum new tokens to generate.")
    args = parser.parse_args()

    MODEL_CONFIG = { 'd_model': 256, 'solidify_chunk_size': 256, 'knn': 5, 'geodesic_step_size': 0.05, 'knn_sampling': 3, 'basename': args.basename, 'learning_rate': 1e-4, 'train_chunk_size': 64 }
    TOKENIZER_CONFIG = {'vocab_size': 8192, 'tokenizer_path': f"{args.basename}_bpe.json"}
    CORPUS_FILE_PATH = f"{args.basename}.corpus.txt"
    TOKEN_FILE_PATH = f"{args.basename}.tokens.bin"
    CAKE_FILE_PATH = f"{args.basename}.cake"
    WEIGHTS_FILE_PATH = f"{args.basename}.weights.pkl"

    print(f"--- WubuMind Funnel Cake Foundry v22.2 (Robust Compact Model) ---")

    if args.command == 'pretokenize':
        print("--- Running Pre-tokenization Step ---")
        if not os.path.exists(CORPUS_FILE_PATH):
            corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
            if not corpora: print("[FATAL] No CORPUS vars found in CORPUS.py."), sys.exit(1)
            print(f"Consolidating CORPUS into a single file: '{CORPUS_FILE_PATH}'...")
            with open(CORPUS_FILE_PATH, 'w', encoding='utf-8') as f:
                for text_chunk in stream_text_from_corpus_data(corpora): f.write(text_chunk + "\n")
        tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
        if not tokenizer.tokenizer: tokenizer.train((line for line in open(CORPUS_FILE_PATH, 'r', encoding='utf-8')), TOKENIZER_CONFIG['vocab_size'])
        print(f"Tokenizing '{CORPUS_FILE_PATH}' and saving to binary file '{TOKEN_FILE_PATH}'...")
        with open(CORPUS_FILE_PATH, 'r', encoding='utf-8') as f_in, open(TOKEN_FILE_PATH, 'wb') as f_out:
            pbar = tqdm(total=os.path.getsize(CORPUS_FILE_PATH), unit='B', unit_scale=True, desc="Tokenizing")
            while chunk := f_in.read(1024 * 1024):
                token_ids = tokenizer.encode(chunk)
                np.array(token_ids, dtype=np.int32).tofile(f_out)
                pbar.update(len(chunk.encode('utf-8')))
            pbar.close()
        print("--- Pre-tokenization complete. You can now run the 'train' command. ---")

    elif args.command == 'train':
        if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file '{TOKEN_FILE_PATH}' not found."), sys.exit(1)
        constructor = FunnelCakeConstructor(MODEL_CONFIG, WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path']))
        if os.path.exists(WEIGHTS_FILE_PATH): print(f"--- Deleting old weights: {WEIGHTS_FILE_PATH} ---"), os.remove(WEIGHTS_FILE_PATH)
        constructor.train(TOKEN_FILE_PATH, epochs=args.epochs, batch_size=args.batch_size)

    elif args.command == 'construct':
        if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file '{TOKEN_FILE_PATH}' not found."), sys.exit(1)
        if not os.path.exists(WEIGHTS_FILE_PATH): print(f"[FATAL] Weights file '{WEIGHTS_FILE_PATH}' not found."), sys.exit(1)
        constructor = FunnelCakeConstructor(MODEL_CONFIG, WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path']))
        if os.path.exists(CAKE_FILE_PATH): print(f"--- Deleting old cake file: {CAKE_FILE_PATH} ---"), os.remove(CAKE_FILE_PATH)
        constructor.construct(TOKEN_FILE_PATH)

    elif args.command == "generate":
        tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
        if not tokenizer.tokenizer or not os.path.exists(CAKE_FILE_PATH): print(f"[FATAL] Model files not found."), sys.exit(1)
        constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)
        constructor.load(args.basename)
        print("\n--- Oracle Command Console (Missile Guidance Edition) ---")
        while True:
            if constructor.should_shutdown: print("\n--- Exiting due to signal. ---"); break
            try: prompt = input("\nYour Prompt> ")
            except EOFError: print("\n--- Exiting. ---"); break
            if prompt.lower() in ["exit", "quit"]: break
            constructor.generate(prompt, max_new=args.max_new, momentum=args.momentum, temp=args.temp)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
        sys.exit(0)
