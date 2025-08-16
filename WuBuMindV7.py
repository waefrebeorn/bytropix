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
from typing import Any, Generator, List, Tuple
import sys
import argparse
from collections import deque
import traceback
import faiss
import signal
import re

# --- Environment Setup ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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

# ... (Helper functions like stream_text_from_corpus_data and WubuTokenizer are unchanged) ...
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
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=True)


class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def project(x):
        # Always project in float32 for stability
        x_f32 = x.astype(jnp.float32)
        norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        max_norm = 1.0 - PoincareBall.EPS
        
        # Perform the projection logic in float32
        projected_f32 = jnp.where(norm_sq >= 1.0, x_f32 / jnp.sqrt(norm_sq).clip(PoincareBall.EPS) * max_norm, x_f32)
        
        # Return in the original dtype
        return projected_f32.astype(x.dtype)

    @staticmethod
    def mobius_add(x, y, c=1.0):
        # Cast inputs to float32 for all calculations
        x_f32, y_f32 = x.astype(jnp.float32), y.astype(jnp.float32)
        
        x2 = jnp.sum(x_f32*x_f32, -1, keepdims=True)
        y2 = jnp.sum(y_f32*y_f32, -1, keepdims=True)
        xy = jnp.sum(x_f32*y_f32, -1, keepdims=True)
        
        num = (1 + 2 * c * xy + c * y2) * x_f32 + (1 - c * x2) * y_f32
        den = 1 + 2 * c * xy + c * c * x2 * y2
        
        # The result of the division is float32, which is what we want for project
        return PoincareBall.project(num / den.clip(PoincareBall.EPS)).astype(x.dtype)

    @staticmethod
    def logmap0(y, c=1.0):
        # Cast input to float32
        y_f32 = y.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True)
        safe_norm = y_norm.clip(PoincareBall.EPS)
        
        # arctanh is very sensitive, so float32 is critical
        arctanh_val = jnp.arctanh(y_norm.clip(max=1.0 - PoincareBall.EPS))
        
        # Calculate result in float32
        result_f32 = jnp.where(safe_norm > 0, arctanh_val * y_f32 / (sqrt_c * safe_norm), jnp.zeros_like(y_f32))
        
        return result_f32 # Return float32 tangent vector

    @staticmethod
    def expmap0(v, c=1.0):
        # Cast input to float32
        v_f32 = v.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True)
        safe_norm = v_norm.clip(PoincareBall.EPS)
        
        # tanh is also sensitive near boundaries
        tanh_val = jnp.tanh(sqrt_c * safe_norm)
        
        # Calculate result in float32
        result_f32 = jnp.where(safe_norm > 0, PoincareBall.project(tanh_val * v_f32 / (sqrt_c * safe_norm)), jnp.zeros_like(v_f32))
        
        return result_f32 # Return float32 point on the ball

    @staticmethod
    def expmap_p(p, v, c=1.0):
        # Cast inputs to float32
        p_f32, v_f32 = p.astype(jnp.float32), v.astype(jnp.float32)
        
        # All intermediate calculations will be float32
        lambda_p = 2. / (1 - c * jnp.sum(p_f32*p_f32, axis=-1, keepdims=True)).clip(PoincareBall.EPS)
        expmapped_v = PoincareBall.expmap0(v_f32 * lambda_p, c) # This will return float32
        
        # mobius_add will handle the final projection and return float32
        return PoincareBall.mobius_add(p_f32, expmapped_v, c).astype(p.dtype)
        
# --- SFIN Inspired Complex-Valued Components ---

class ComplexEmbedding(nn.Module):
    vocab_size: int
    features: int # Note: features is for each component (real/imag)
    dtype: Any
    
    @nn.compact
    def __call__(self, x):
        real_embed = nn.Embed(self.vocab_size, self.features, name="real_embed", dtype=self.dtype)(x)
        imag_embed = nn.Embed(self.vocab_size, self.features, name="imag_embed", dtype=self.dtype)(x)
        return real_embed, imag_embed

class ComplexGRUCell(nn.Module):
    features: int
    dtype: Any

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x: Tuple[jnp.ndarray, jnp.ndarray]):
        h_r, h_i = carry
        x_r, x_i = x

        # --- Correct Flax Pattern: Define all layers once at the top ---
        # Update Gate Weights (z_gate)
        h_to_zr = nn.Dense(self.features, use_bias=False, name="h_to_zr", dtype=self.dtype)
        h_to_zi = nn.Dense(self.features, use_bias=False, name="h_to_zi", dtype=self.dtype)
        x_to_zr = nn.Dense(self.features, use_bias=False, name="x_to_zr", dtype=self.dtype)
        x_to_zi = nn.Dense(self.features, use_bias=False, name="x_to_zi", dtype=self.dtype)

        # Reset Gate Weights (r_gate)
        h_to_rr = nn.Dense(self.features, use_bias=False, name="h_to_rr", dtype=self.dtype)
        h_to_ri = nn.Dense(self.features, use_bias=False, name="h_to_ri", dtype=self.dtype)
        x_to_rr = nn.Dense(self.features, use_bias=False, name="x_to_rr", dtype=self.dtype)
        x_to_ri = nn.Dense(self.features, use_bias=False, name="x_to_ri", dtype=self.dtype)

        # Candidate Hidden State Weights (n_gate)
        h_to_nr = nn.Dense(self.features, use_bias=False, name="h_to_nr", dtype=self.dtype)
        h_to_ni = nn.Dense(self.features, use_bias=False, name="h_to_ni", dtype=self.dtype)
        x_to_nr = nn.Dense(self.features, use_bias=False, name="x_to_nr", dtype=self.dtype)
        x_to_ni = nn.Dense(self.features, use_bias=False, name="x_to_ni", dtype=self.dtype)
        # --- End of layer definitions ---

        # Helper to create parameters with the correct dtype
        def get_bias(name):
            return self.param(name, nn.initializers.zeros, (self.features,), self.dtype)

        # Simulating complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i

        # Update gate
        z_r_linear = h_to_zr(h_r) - h_to_zi(h_i) + x_to_zr(x_r) - x_to_zi(x_i)
        z_i_linear = h_to_zr(h_i) + h_to_zi(h_r) + x_to_zr(x_i) + x_to_zi(x_r)
        z_r = nn.sigmoid(z_r_linear + get_bias('z_bias_r'))
        z_i = nn.sigmoid(z_i_linear + get_bias('z_bias_i'))
        
        # Reset gate
        r_r_linear = h_to_rr(h_r) - h_to_ri(h_i) + x_to_rr(x_r) - x_to_ri(x_i)
        r_i_linear = h_to_rr(h_i) + h_to_ri(h_r) + x_to_rr(x_i) + x_to_ri(x_r)
        r_r = nn.sigmoid(r_r_linear + get_bias('r_bias_r'))
        r_i = nn.sigmoid(r_i_linear + get_bias('r_bias_i'))
        
        # Candidate hidden state
        rh_r, rh_i = r_r * h_r - r_i * h_i, r_r * h_i + r_i * h_r
        
        n_r_linear = h_to_nr(rh_r) - h_to_ni(rh_i) + x_to_nr(x_r) - x_to_ni(x_i)
        n_i_linear = h_to_nr(rh_i) + h_to_ni(rh_r) + x_to_nr(x_i) + x_to_ni(x_r)
        n_r = jnp.tanh(n_r_linear + get_bias('n_bias_r'))
        n_i = jnp.tanh(n_i_linear + get_bias('n_bias_i'))
        
        # Final hidden state update with interference: h' = (1-z)*h + z*n
        one_minus_z_r, one_minus_z_i = 1 - z_r, -z_i
        
        term1_r = one_minus_z_r * h_r - one_minus_z_i * h_i
        term1_i = one_minus_z_r * h_i + one_minus_z_i * h_r
        
        term2_r = z_r * n_r - z_i * n_i
        term2_i = z_r * n_i + z_i * n_r
        
        new_h_r = term1_r + term2_r
        new_h_i = term1_i + term2_i
        
        return (new_h_r, new_h_i), (new_h_r, new_h_i)
        
class WaveFunctionCollapseHead(nn.Module):
    vocab_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, h_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        h_r, h_i = h_complex
        
        # Project real and imaginary parts to vocab size to get complex logits
        real_logits = nn.Dense(self.vocab_size, name="real_collapse", dtype=self.dtype)(h_r)
        imag_logits = nn.Dense(self.vocab_size, name="imag_collapse", dtype=self.dtype)(h_i)
        
        # Born rule: probability is the squared magnitude of the complex amplitude
        # We use squared magnitude for logits to maintain numerical stability before softmax
        final_logits = real_logits**2 + imag_logits**2
        return final_logits

# --- Main Model Architecture ---

class DripHead(nn.Module):
    d_model_total: int # This will be split between real and imag
    vocab_size: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, token_ids, initial_carry_complex):
        d_model_comp = self.d_model_total // 2
        
        # 1. Embed tokens into complex space
        token_embeds_r, token_embeds_i = ComplexEmbedding(self.vocab_size, d_model_comp, dtype=self.dtype, name="token_embed")(token_ids)
        
        # 2. Scan with ComplexGRU
        scanner = nn.scan(
            ComplexGRUCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0
        )
        final_carry_complex, hidden_states_complex = scanner(features=d_model_comp, dtype=self.dtype)(initial_carry_complex, (token_embeds_r, token_embeds_i))
        
        # 3. Collapse wave function to get logits
        logits = WaveFunctionCollapseHead(vocab_size=self.vocab_size)(hidden_states_complex)
        
        return final_carry_complex, hidden_states_complex, logits

class FunnelCakeConstructor:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        self.d_model = config['d_model']
        self.key = jax.random.PRNGKey(42)
        
        self.drip_head = DripHead(d_model_total=self.d_model, vocab_size=tokenizer.get_vocab_size())
        self.train_state = None; self.drip_head_params = None
        self.formation_space = deque()
        
        # H_sphere_points are now in a real-valued space of size d_model
        self.H_sphere_points = np.empty((0, self.d_model), dtype=np.float32)
        self.H_sphere_metadata = []; self.hyperbolic_index = None
        
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if not self.should_shutdown:
            print("\n--- SIGINT received. Saving/exiting when possible. Press again to force quit. ---")
            self.should_shutdown = True
        else:
            print("\n--- Forced exit. ---"); sys.exit(1)

    def _init_drip_head(self, training=False):
        if self.drip_head_params is not None or self.train_state is not None: return
        mode_str = "for Training" if training else "(Random Projection Mode)"
        print(f"--- Initializing Complex Drip Head {mode_str}... ---")
        self.key, drip_key = jax.random.split(self.key)
        
        dummy_tokens = jnp.zeros((16,), dtype=jnp.int32)
        d_model_comp = self.d_model // 2
        dummy_hidden_complex = (
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype),
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype)
        )
        
        params = self.drip_head.init(drip_key, dummy_tokens, dummy_hidden_complex)['params']
        param_count = sum(x.size for x in jax.tree.leaves(params))
        
        if training:
            tx = optax.adamw(self.config['learning_rate'])
            self.train_state = train_state.TrainState.create(apply_fn=self.drip_head.apply, params=params, tx=tx)
            print(f"--- Drip Head Initialized for Training: {param_count:,} params. ---")
        else:
            self.drip_head_params = params
            print(f"--- Drip Head Initialized: {param_count:,} params. ---")
    
    def train(self, corpus_filepath, steps=5000):
        self._init_drip_head(training=True)
        chunk_size = self.config['train_chunk_size']
        d_model_comp = self.d_model // 2
        
        def data_generator():
            with open(corpus_filepath, 'r', encoding='utf-8', errors='ignore') as f: text = f.read()
            tokens = self.tokenizer.encode(text); n_tokens = len(tokens)
            while True:
                start1 = np.random.randint(0, n_tokens - chunk_size * 2)
                anchor = tokens[start1 : start1 + chunk_size]; positive = tokens[start1 + 10 : start1 + 10 + chunk_size]
                start2 = np.random.randint(0, n_tokens - chunk_size)
                negative = tokens[start2 : start2 + chunk_size]
                yield jnp.array(anchor), jnp.array(positive), jnp.array(negative)

        @jax.jit
        def train_step(state, batch):
            anchor, positive, negative = batch
            def loss_fn(params):
                initial_hidden_complex = (
                    jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype),
                    jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype)
                )
                _, anchor_h_c, _ = state.apply_fn({'params': params}, anchor, initial_hidden_complex)
                _, positive_h_c, _ = state.apply_fn({'params': params}, positive, initial_hidden_complex)
                _, negative_h_c, _ = state.apply_fn({'params': params}, negative, initial_hidden_complex)
                
                # Final complex hidden states
                ah_r, ah_i = anchor_h_c[0][-1], anchor_h_c[1][-1]
                ph_r, ph_i = positive_h_c[0][-1], positive_h_c[1][-1]
                nh_r, nh_i = negative_h_c[0][-1], negative_h_c[1][-1]
                
                margin = 0.5
                
                # Squared Euclidean distance in complex space
                dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2)
                dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2)
                
                loss = jnp.maximum(0, dist_pos - dist_neg + margin)
                return loss
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        data_gen = data_generator()
        pbar = tqdm(range(steps), desc="Cheater Training (SFIN)")
        for step in pbar:
            batch = next(data_gen)
            self.train_state, loss = train_step(self.train_state, batch)
            if step % 50 == 0: pbar.set_postfix(loss=f"{loss.item():.4f}")
            if self.should_shutdown: print("\n--- Interrupt honored. Saving trained weights... ---"); break
        
        self.drip_head_params = self.train_state.params
        self.save_trained_weights(self.config['basename'])

    def save_trained_weights(self, basename):
        params_to_save = self.drip_head_params
        if self.train_state is not None: params_to_save = self.train_state.params
        if params_to_save is None: print("[WARN] No Drip Head parameters found to save."); return
        print(f"--- Saving Drip Head weights to {basename}.weights.pkl ---")
        with open(f"{basename}.weights.pkl", 'wb') as f: pickle.dump(jax.device_get(params_to_save), f)
        print("--- Weights saved. ---")
    
    def load_trained_weights(self, basename):
        weights_file = f"{basename}.weights.pkl"
        self._init_drip_head(training=False)
        if not os.path.exists(weights_file):
            print("[INFO] No pre-trained weights found. Using random projection.")
            return
        
        print(f"--- Loading trained Drip Head weights from {weights_file} ---")
        with open(weights_file, 'rb') as f: self.drip_head_params = pickle.load(f)
        param_count = sum(x.size for x in jax.tree.leaves(self.drip_head_params))
        print(f"--- Trained weights loaded. {param_count:,} params. ---")
    
    def construct(self, corpus_filepath):
        self.should_shutdown = False
        self.load_trained_weights(self.config['basename'])
        
        print(f"--- Constructing Funnel Cake from '{corpus_filepath}'... ---")
        batch_size = 256
        d_model_comp = self.d_model // 2

        @jax.jit
        def drip_batch_step(params, token_batch, initial_hidden_complex):
            final_hidden_complex, all_hidden_complex, _ = self.drip_head.apply({'params': params}, token_batch, initial_hidden_complex)
            return final_hidden_complex, all_hidden_complex

        hidden_state_complex = (
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype),
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype)
        )
        file_size = os.path.getsize(corpus_filepath)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Constructing Filament")
        last_update_time = time.time(); tokens_since_last_update = 0

        with open(corpus_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            while (text_chunk := f.read(128 * 1024)):
                token_ids = self.tokenizer.encode(text_chunk)
                for i in range(0, len(token_ids), batch_size):
                    batch = jnp.array(token_ids[i:i+batch_size], dtype=jnp.int32)
                    if len(batch) == 0: continue
                    
                    final_hidden_complex, batch_hidden_states_complex = drip_batch_step(self.drip_head_params, batch, hidden_state_complex)
                    hidden_state_complex = final_hidden_complex
                    
                    batch_cpu = np.array(batch)
                    # Collapse complex states to real-valued vectors for formation space
                    hidden_states_r = np.array(batch_hidden_states_complex[0])
                    hidden_states_i = np.array(batch_hidden_states_complex[1])
                    hidden_states_collapsed = np.concatenate([hidden_states_r, hidden_states_i], axis=-1)

                    for j in range(len(batch_cpu)):
                        self.formation_space.append({'drip_head_state': hidden_states_collapsed[j], 'token_id': batch_cpu[j].item()})
                    
                    while len(self.formation_space) >= self.config['solidify_chunk_size']: self._solidify()
                    tokens_since_last_update += len(batch_cpu)
                
                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed > 1.0:
                    rate = tokens_since_last_update / elapsed
                    pbar.set_postfix(solids=f"{len(self.H_sphere_metadata)}", rate=f"{rate:,.0f} tok/s")
                    last_update_time, tokens_since_last_update = current_time, 0
                pbar.update(len(text_chunk.encode('utf-8')))
                if self.should_shutdown: print("\n--- Interrupt honored. Finishing up... ---"); break
        
        if self.formation_space: self._solidify(force_all=True)
        pbar.close()
        print(f"\n--- Construction Complete. Solids: {self.H_sphere_points.shape[0]} ---")
        self.save(self.config['basename'])

    def _solidify(self, force_all=False):
        chunk_size = len(self.formation_space) if force_all else self.config['solidify_chunk_size']
        if chunk_size == 0: return
        chunk = [self.formation_space.popleft() for _ in range(chunk_size)]
        
        # Drip states are now real-valued collapsed states
        drip_states = np.array([item['drip_head_state'] for item in chunk]).astype(np.float32)
        tangent_vec = np.mean(drip_states, axis=0)

        # The rest of this function remains the same as it operates on real vectors
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
        # ... (save function is unchanged and correct) ...
        print(f"--- Saving Funnel Cake to {basename}.cake ---")
        state = {'config': self.config, 'H_sphere_points': self.H_sphere_points, 'H_sphere_metadata': self.H_sphere_metadata}
        with open(f"{basename}.cake", 'wb') as f: pickle.dump(state, f)
        self.save_trained_weights(basename)
        print("--- Save complete. ---")

    def load(self, basename):
        # ... (load function is unchanged and correct) ...
        print(f"--- Loading Funnel Cake from {basename}.cake ---")
        with open(f"{basename}.cake", 'rb') as f: state = pickle.load(f)
        self.config = state['config']
        self.load_trained_weights(basename)
        self.H_sphere_points = state['H_sphere_points']; self.H_sphere_metadata = state['H_sphere_metadata']
        if self.H_sphere_points.shape[0] > 0:
            self.hyperbolic_index = faiss.IndexFlatL2(self.d_model)
            self.hyperbolic_index.add(self.H_sphere_points.astype(np.float32))
        print(f"--- Funnel Cake loaded. Contains {self.H_sphere_points.shape[0]} solidified points. ---")

    def generate(self, prompt, max_new=200, momentum=0.8, temp=0.8):
        self.should_shutdown = False
        if self.hyperbolic_index is None: print("\n[ERROR] Funnel Cake is empty."), sys.exit(1)
        print(f"\n\033[1;32m{prompt}\033[0m", end=''); sys.stdout.flush()
        
        d_model_comp = self.d_model // 2

        @jax.jit
        def get_logits_and_state(params, tokens, hidden_complex):
            final_hidden_complex, _, logits = self.drip_head.apply({'params': params}, tokens, hidden_complex)
            return final_hidden_complex, logits

        hidden_state_complex = (
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype),
            jnp.zeros((d_model_comp,), dtype=self.drip_head.dtype)
        )
        
        prompt_tokens = self.tokenizer.encode(prompt)
        if prompt_tokens:
             hidden_state_complex, _ = get_logits_and_state(self.drip_head_params, jnp.array(prompt_tokens), hidden_state_complex)
        
        # Collapse the hidden state to a real vector for manifold interaction
        hidden_state_collapsed = jnp.concatenate(hidden_state_complex)
        
        # --- MISSILE GUIDANCE SYSTEM START ---
        
        # 1. ACQUIRE INITIAL TARGET: Find the closest point on the manifold to our current context
        _, start_indices = self.hyperbolic_index.search(np.array(hidden_state_collapsed, dtype=np.float32).reshape(1, -1), 1)
        current_point = jnp.array(self.H_sphere_points[start_indices[0][0]])
        velocity_vector = jnp.zeros_like(current_point)
        
        for _ in range(max_new):
            # 2. CALCULATE DEVIATION: The difference between where the context is pointing and where we are.
            # This is the primary corrective command for the missile.
            intent_vector = PoincareBall.logmap0(current_point) - hidden_state_collapsed
            
            # 3. GET LOCAL READINGS: Poll neighbors on the manifold to understand the local semantic flow.
            _, neighbor_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), self.config['knn_sampling'])
            neighbors = jnp.array(self.H_sphere_points[neighbor_indices.flatten()])
            tangent_vectors = jax.vmap(PoincareBall.logmap0)(neighbors)
            local_flow_vector = jnp.mean(tangent_vectors, axis=0)
            
            # 4. GENERATE CORRECTIVE COMMANDS: Combine intent, momentum, and local flow into a new velocity.
            new_velocity = (velocity_vector * momentum) + (intent_vector * 0.2) + (local_flow_vector * (1 - momentum))
            velocity_vector = new_velocity / jnp.linalg.norm(new_velocity).clip(1e-6)

            # 5. DRIVE THE MISSILE: Move along the geodesic in the direction of the new velocity.
            current_point = PoincareBall.expmap_p(current_point, velocity_vector * self.config['geodesic_step_size'])
            
            # --- MISSILE GUIDANCE SYSTEM END ---

            # Find the semantic chunk at our new position
            _, closest_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), 1)
            chosen_neighbor_idx = closest_indices.flatten()[0]
            retrieved_chunk_tokens = self.H_sphere_metadata[chosen_neighbor_idx]['token_ids']
            if not retrieved_chunk_tokens: continue
            
            # Use the Drip Head to generate the next token, constrained by the retrieved chunk
            # We use a dummy token [1] to advance the state by one step to get next-step predictions
            _, logits = get_logits_and_state(self.drip_head_params, jnp.array([1]), hidden_state_complex)
            next_step_logits = logits.squeeze(0) 

            allowed_tokens = jnp.array(list(set(retrieved_chunk_tokens)))
            mask = jnp.full(self.tokenizer.get_vocab_size(), -jnp.inf)
            mask = mask.at[allowed_tokens].set(0.0)

            masked_logits = next_step_logits + mask
            probs = nn.softmax(masked_logits / temp)
            self.key, subkey = jax.random.split(self.key)
            next_token_id = jax.random.categorical(subkey, jnp.log(probs.clip(1e-9)))

            print(self.tokenizer.decode([next_token_id.item()]), end=''); sys.stdout.flush()
            
            # Update the complex hidden state with the generated token
            hidden_state_complex, _ = get_logits_and_state(self.drip_head_params, jnp.array([next_token_id]), hidden_state_complex)
            hidden_state_collapsed = jnp.concatenate(hidden_state_complex)
            
            if self.should_shutdown: print("\n--- Interrupt honored. ---"); break
        print()

def main():
    parser = argparse.ArgumentParser(description="WubuMind Funnel Cake Constructor v9.0 (SFIN Integration)")
    parser.add_argument('command', choices=['train', 'construct', 'generate'], help="The command to execute.")
    parser.add_argument('--basename', type=str, default="wubumind_funnel_cake_v1", help="Basename for model files.")
    args = parser.parse_args()

    # NOTE: d_model MUST be even for the complex representation split.
    MODEL_CONFIG = {'d_model': 256, 'solidify_chunk_size': 256, 'knn': 5, 'geodesic_step_size': 0.05, 'knn_sampling': 3, 'basename': args.basename, 'learning_rate': 1e-4, 'train_chunk_size': 64}
    TOKENIZER_CONFIG = {'vocab_size': 4096, 'tokenizer_path': f"{args.basename}_bpe.json"}
    CORPUS_FILE_PATH = f"{args.basename}.corpus.txt"
    CAKE_FILE_PATH = f"{args.basename}.cake"
    WEIGHTS_FILE_PATH = f"{args.basename}.weights.pkl"

    print(f"--- WubuMind Funnel Cake Foundry v9.0 (SFIN Integration) ---")
    
    # ... The rest of main() function remains the same as the corrected version from before ...
    if args.command == 'train':
        print("--- Starting full build process: Training then Construction ---")
        if not os.path.exists(CORPUS_FILE_PATH):
            corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
            if not corpora: print("[FATAL] No CORPUS vars found in CORPUS.py."), sys.exit(1)
            print(f"Consolidating CORPUS into a single file: '{CORPUS_FILE_PATH}'...")
            with open(CORPUS_FILE_PATH, 'w', encoding='utf-8') as f:
                for text_chunk in stream_text_from_corpus_data(corpora): f.write(text_chunk + "\n")
            print("Corpus consolidation complete.")
        
        tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
        if not tokenizer.tokenizer:
            tokenizer.train((line for line in open(CORPUS_FILE_PATH, 'r', encoding='utf-8')), TOKENIZER_CONFIG['vocab_size'])

        constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)
        if os.path.exists(WEIGHTS_FILE_PATH):
            print(f"--- Deleting old weights file before training: {WEIGHTS_FILE_PATH} ---")
            os.remove(WEIGHTS_FILE_PATH)
        constructor.train(CORPUS_FILE_PATH, steps=10000)

        print("\n--- Training complete. Automatically proceeding to construction phase. ---")
        if os.path.exists(CAKE_FILE_PATH):
            print(f"--- Deleting old cake file to build a new one: {CAKE_FILE_PATH} ---")
            os.remove(CAKE_FILE_PATH)
        constructor.construct(CORPUS_FILE_PATH)
        print("\n--- Full build process complete. Model is ready for generation. ---")

    elif args.command == 'construct':
        print("--- Running standalone construction process. ---")
        if not os.path.exists(WEIGHTS_FILE_PATH):
            print(f"[FATAL] Weights file not found at '{WEIGHTS_FILE_PATH}'. Please run the 'train' command first.")
            sys.exit(1)
        
        tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
        if not tokenizer.tokenizer:
             print(f"[FATAL] Tokenizer not found at '{TOKENIZER_CONFIG['tokenizer_path']}'. Please run 'train' first to generate it.")
             sys.exit(1)

        constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)
        if os.path.exists(CAKE_FILE_PATH):
            print(f"--- Deleting old cake file: {CAKE_FILE_PATH} ---")
            os.remove(CAKE_FILE_PATH)
        constructor.construct(CORPUS_FILE_PATH)

    elif args.command == "generate":
        if not os.path.exists(TOKENIZER_CONFIG['tokenizer_path']) or not os.path.exists(CAKE_FILE_PATH):
            print(f"[FATAL] Model files not found. Please run 'train' first.")
            sys.exit(1)
        
        tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
        constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)
        constructor.load(args.basename)

        print("\n--- Oracle Command Console (Missile Guidance Edition) ---")
        while True:
            if constructor.should_shutdown:
                print("\n--- Exiting due to signal. ---"); break
            try:
                prompt = input("\nYour Prompt> ")
            except EOFError:
                print("\n--- Exiting. ---"); break
            if prompt.lower() in ["exit", "quit"]: break
            constructor.generate(prompt)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
        sys.exit(0)
