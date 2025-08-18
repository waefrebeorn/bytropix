import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from tqdm import tqdm
import pickle
from typing import Any, Generator, Tuple, Dict
import sys
import argparse
from collections import deque, Counter
import signal
from dataclasses import dataclass
from functools import partial

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
# --- Import BallTree for Spatial Partitioning ---
try:
    from sklearn.neighbors import BallTree
except ImportError: print("[FATAL] `scikit-learn` not found. `pip install scikit-learn`."), sys.exit(1)

# --- XJDR's Metacognitive Sampler Logic (Integrated) ---

@dataclass(frozen=True)
class SamplerConfig:
  low_entropy_threshold = 0.3
  high_entropy_threshold = 2.5
  low_varentropy_threshold = 1.2
  high_varentropy_threshold = 2.5
  clarifying_question_token: int = 2 # Default to <UNK> if <CQ> isn't found

def get_entropy_metrics(logits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    varentropy = jnp.var(log_probs, axis=-1)
    return entropy, varentropy

@partial(jax.jit, static_argnames=("config",))
def xjdr_metacognitive_sample(
  key: jax.random.PRNGKey,
  logits: jnp.ndarray,
  config: SamplerConfig,
) -> jnp.ndarray:
    def _and(*args):
        return jnp.all(jnp.array(args))

    entropy, varentropy = get_entropy_metrics(logits)
    is_lelv = _and(entropy < config.low_entropy_threshold, varentropy < config.low_varentropy_threshold)
    is_helv = _and(entropy > config.high_entropy_threshold, varentropy < config.low_varentropy_threshold)
    is_lehv = _and(entropy < config.high_entropy_threshold, varentropy > config.high_varentropy_threshold)
    is_hehv = _and(entropy > config.high_entropy_threshold, varentropy > config.high_varentropy_threshold)
    case_index = jnp.argmax(jnp.array([is_lelv, is_helv, is_lehv, is_hehv, True]))
    
    def lelv_case(): return jax.random.categorical(key, logits)
    def helv_case(): return jnp.array(config.clarifying_question_token, dtype=jnp.int32)
    def lehv_case(): return jax.random.categorical(key, logits)
    def hehv_case():
        first_token = jax.random.categorical(key, logits)
        penalized_logits = logits.at[first_token].set(-jnp.inf)
        return jax.random.categorical(key, penalized_logits)
    def default_case(): return jax.random.categorical(key, logits)
        
    branches = [lelv_case, helv_case, lehv_case, hehv_case, default_case]
    return jax.lax.switch(case_index, branches)

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
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<PAD>", "<UNK>", "<CQ>"])
        self.tokenizer.train_from_iterator(corpus_iterator, trainer)
        self.tokenizer.save(self.tokenizer_path)
        print(f"--- Tokenizer trained. Vocab: {self.get_vocab_size()}. Saved to {self.tokenizer_path} ---")
    def get_vocab_size(self): return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    def encode(self, text): return self.tokenizer.encode(text).ids if self.tokenizer else []
    def decode(self, ids):
        if not self.tokenizer: return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

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

# --- Main Model Architectures ---

# --- 1. The Navigator (DripHead) ---
class GRUCell(nn.Module):
    d_model_total: int; dtype: Any
    @nn.compact
    def __call__(self, carry, x):
        xh = jnp.concatenate([x, carry], axis=-1)
        r = nn.sigmoid(nn.Dense(self.d_model_total, name="reset_gate_d", dtype=self.dtype)(xh))
        u = nn.sigmoid(nn.Dense(self.d_model_total, name="update_gate_d", dtype=self.dtype)(xh))
        c_in = jnp.concatenate([x, r * carry], axis=-1)
        c = nn.tanh(nn.Dense(self.d_model_total, name="candidate_gate_d", dtype=self.dtype)(c_in))
        new_carry = (1 - u) * carry + u * c
        return new_carry, new_carry

class DripHead(nn.Module):
    d_model_total: int; vocab_size: int; dtype: Any = jnp.bfloat16
    def setup(self):
        self.d_model_comp = self.d_model_total // 2
        self.token_embed = ComplexEmbedding(self.vocab_size, self.d_model_comp, name="token_embed", dtype=self.dtype)
        self.layer_norm = ComplexLayerNorm(dtype=self.dtype, name="layer_norm")
        scanner = nn.scan(
            GRUCell, variable_broadcast='params', split_rngs={'params': False},
            in_axes=1, out_axes=1
        )
        self.gru = scanner(d_model_total=self.d_model_total, dtype=self.dtype)
    def __call__(self, token_ids):
        batch_size = token_ids.shape[0]
        initial_carry_real = jnp.zeros((batch_size, self.d_model_total), dtype=self.dtype)
        token_embeds_r, token_embeds_i = self.token_embed(token_ids)
        xs_real = jnp.concatenate([token_embeds_r, token_embeds_i], axis=-1)
        _, hidden_states_real = self.gru(initial_carry_real, xs_real)
        h_r, h_i = jnp.split(hidden_states_real, 2, axis=-1)
        return self.layer_norm((h_r, h_i))

# --- 2. The Oracle (WubuSphere) ---
class WubuSphere(nn.Module):
    vocab_size: int; d_model: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, h_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        h_r, h_i = h_complex
        hidden_state_collapsed = jnp.concatenate([h_r, h_i], axis=-1)
        x = nn.Dense(self.d_model * 2, name="oracle_hidden", dtype=self.dtype)(hidden_state_collapsed)
        x = nn.gelu(x)
        return nn.Dense(self.vocab_size, name="oracle_output", dtype=self.dtype)(x)

# --- The Funnel Cake System ---
class FunnelCakeConstructor:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        self.d_model = config['d_model']
        self.key = jax.random.PRNGKey(42)
        self.navigator = DripHead(d_model_total=self.d_model, vocab_size=tokenizer.get_vocab_size())
        self.oracle = WubuSphere(vocab_size=tokenizer.get_vocab_size(), d_model=self.d_model)
        self.nav_train_state = None; self.nav_params = None
        self.oracle_train_state = None; self.oracle_params = None
        self.formation_space = deque()
        self.H_sphere_points = []; self.H_sphere_metadata = []; self.ball_tree = None
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        print("\n--- SIGINT received. Shutting down gracefully. ---")
        self.should_shutdown = True

    def _init_models(self, training_mode: str):
        print(f"--- Initializing/Configuring Models for {training_mode}... ---")
        self.key, nav_key, oracle_key = jax.random.split(self.key, 3)
        dummy_tokens = jnp.zeros((2, self.config['train_chunk_size']), dtype=jnp.int32)
        
        if self.nav_params is None:
            print("...Initializing new Navigator from scratch.")
            self.nav_params = self.navigator.init(nav_key, dummy_tokens)['params']

        if self.oracle_params is None:
            print("...Initializing new Oracle from scratch.")
            dummy_hidden_state = self.navigator.apply({'params': self.nav_params}, dummy_tokens)
            self.oracle_params = self.oracle.init(oracle_key, dummy_hidden_state)['params']

        if training_mode == "navigator":
            print("...Configuring TrainState for Navigator.")
            tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.config['learning_rate']))
            self.nav_train_state = train_state.TrainState.create(
                apply_fn=self.navigator.apply, params=self.nav_params, tx=tx)
        elif training_mode == "oracle":
            print("...Configuring TrainState for Oracle.")
            tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.config['learning_rate_oracle']))
            self.oracle_train_state = train_state.TrainState.create(
                apply_fn=self.oracle.apply, params=self.oracle_params, tx=tx)

        nav_pcount = sum(x.size for x in jax.tree.leaves(self.nav_params))
        ora_pcount = sum(x.size for x in jax.tree.leaves(self.oracle_params))
        print(f"--- Model Config Complete. Navigator: {nav_pcount:,} params | Oracle: {ora_pcount:,} params. ---")
        
    def train_navigator(self, token_file_path, epochs=3, batch_size=4096):
        self._init_models(training_mode="navigator")
        chunk_size = self.config['train_chunk_size']
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')
        indices = np.arange(0, len(tokens) - (chunk_size + 10))

        @jax.jit
        def train_step(state, batch, margin):
            anchor, positive, negative = batch
            def loss_fn(params):
                anchor_h_c, positive_h_c, negative_h_c = [state.apply_fn({'params': params}, x) for x in (anchor, positive, negative)]
                ah_r, ah_i = anchor_h_c[0][:, -1, :], anchor_h_c[1][:, -1, :]
                ph_r, ph_i = positive_h_c[0][:, -1, :], positive_h_c[1][:, -1, :]
                nh_r, nh_i = negative_h_c[0][:, -1, :], negative_h_c[1][:, -1, :]
                dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2, axis=-1)
                dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2, axis=-1)
                return jnp.mean(jnp.maximum(0, dist_pos - dist_neg + margin))
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss
        
        for epoch in range(epochs):
            if self.should_shutdown: break
            margin = 0.25 + (epoch / max(1, epochs-1)) * 0.75
            print(f"\n--- Starting Navigator Epoch {epoch + 1}/{epochs} (Margin: {margin:.2f}) ---")
            np.random.shuffle(indices)
            pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}", total=len(indices)//batch_size)
            for i in pbar:
                if self.should_shutdown: break
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size: continue
                
                starts = {'anchor': batch_indices, 'positive': batch_indices + 10, 'negative': np.random.randint(0, len(tokens) - chunk_size, size=len(batch_indices))}
                batch_data = {k: np.array([tokens[s:s+chunk_size] for s in v]) for k, v in starts.items()}
                
                self.nav_train_state, loss = train_step(self.nav_train_state, tuple(batch_data.values()), margin)
                pbar.set_postfix(avg_loss=f"{loss.item():.4f}")
        
        if not self.should_shutdown: print("\n--- Navigator Training Complete ---")
        self.nav_params = self.nav_train_state.params
        self.save_weights(self.config['basename'])

    def train_oracle(self, token_file_path, epochs=5, batch_size=896):
        print("--- Loading pre-trained Navigator for Oracle training... ---")
        self.load_weights(self.config['basename'])
        self._init_models(training_mode="oracle")
        
        chunk_size = self.config['train_chunk_size']
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')
        indices = np.arange(0, len(tokens) - chunk_size)
        num_steps_per_epoch = len(indices) // batch_size
        total_steps = epochs * num_steps_per_epoch

        # --- Use Advanced Optimizer for Oracle ---
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=self.config['learning_rate_oracle'], warmup_steps=int(total_steps * 0.1), decay_steps=total_steps)
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule))
        self.oracle_train_state = train_state.TrainState.create(apply_fn=self.oracle.apply, params=self.oracle_params, tx=tx)

        @jax.jit
        def train_step(nav_params, oracle_state, batch):
            inputs, labels = batch
            def loss_fn(oracle_params):
                hidden_states = self.navigator.apply({'params': nav_params}, inputs)
                logits = self.oracle.apply({'params': oracle_params}, hidden_states)
                one_hot_labels = jax.nn.one_hot(labels, num_classes=self.tokenizer.get_vocab_size())
                return optax.softmax_cross_entropy(logits, one_hot_labels).mean()
            loss, grads = jax.value_and_grad(loss_fn)(oracle_state.params)
            return oracle_state.apply_gradients(grads=grads), loss
        
        for epoch in range(epochs):
            if self.should_shutdown: break
            print(f"\n--- Starting Oracle Epoch {epoch + 1}/{epochs} ---")
            np.random.shuffle(indices)
            pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}", total=len(indices)//batch_size)
            for i in pbar:
                if self.should_shutdown: break
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size: continue
                
                input_batch = np.array([tokens[s:s+chunk_size-1] for s in batch_indices])
                label_batch = np.array([tokens[s+1:s+chunk_size] for s in batch_indices])
                
                self.oracle_train_state, loss = train_step(self.nav_params, self.oracle_train_state, (input_batch, label_batch))
                pbar.set_postfix(avg_loss=f"{loss.item():.4f}")

        if not self.should_shutdown: print("\n--- Oracle Training Complete ---")
        self.oracle_params = self.oracle_train_state.params
        self.save_weights(self.config['basename'])

    def save_weights(self, basename):
        if self.nav_params:
            print(f"--- Saving Navigator weights to {basename}.nav.pkl ---")
            with open(f"{basename}.nav.pkl", 'wb') as f: pickle.dump(jax.device_get(self.nav_params), f)
        if self.oracle_params:
            print(f"--- Saving Oracle weights to {basename}.oracle.pkl ---")
            with open(f"{basename}.oracle.pkl", 'wb') as f: pickle.dump(jax.device_get(self.oracle_params), f)
        print("--- Weights saved. ---")

    def load_weights(self, basename):
        nav_file, oracle_file = f"{basename}.nav.pkl", f"{basename}.oracle.pkl"
        if os.path.exists(nav_file):
            print(f"--- Loading Navigator weights from {nav_file} ---")
            with open(nav_file, 'rb') as f: self.nav_params = pickle.load(f)
        if os.path.exists(oracle_file):
            print(f"--- Loading Oracle weights from {oracle_file} ---")
            with open(oracle_file, 'rb') as f: self.oracle_params = pickle.load(f)

    def construct(self, token_file_path, batch_size=512):
        self.load_weights(self.config['basename'])
        if self.nav_params is None: print("[FATAL] Navigator must be trained before construction."), sys.exit(1)
        print(f"--- Constructing Funnel Cake from memory-mapped tokens... ---")
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')

        @jax.jit
        def get_hidden_states(params, token_batch):
            h_r, h_i = self.navigator.apply({'params': params}, token_batch)
            return jnp.concatenate([h_r, h_i], axis=-1)

        # *** NEW: Track the global token offset for metadata ***
        current_token_offset = 0
        pbar = tqdm(total=len(tokens), unit='tok', unit_scale=True, desc="Constructing")
        for i in range(0, len(tokens), batch_size):
            if self.should_shutdown: break
            batch_tokens = tokens[i:i+batch_size]
            if len(batch_tokens) == 0: continue
            
            states = get_hidden_states(self.nav_params, jnp.array([batch_tokens]))
            for j in range(len(batch_tokens)):
                # *** NEW: Pass the global index along with the state ***
                self.formation_space.append({
                    'drip_head_state': states[0, j],
                    'token_id': batch_tokens[j].item(),
                    'start_token_idx': current_token_offset + j
                })
            
            while len(self.formation_space) >= self.config['solidify_chunk_size']: self._solidify()
            current_token_offset += len(batch_tokens) # Update offset
            pbar.update(len(batch_tokens))
            pbar.set_postfix(solids=f"{len(self.H_sphere_points):,}")
        
        if self.formation_space: self._solidify(force_all=True)
        pbar.close()

        print(f"\n--- Construction Complete. Total Solids: {len(self.H_sphere_points)}. Building Ball Tree... ---")
        if self.H_sphere_points:
            self.ball_tree = BallTree(np.array(self.H_sphere_points, dtype=np.float32), leaf_size=40)
            print(f"--- Ball Tree constructed. ---")
        self.save_cake(self.config['basename'])

    def _solidify(self, force_all=False):
        chunk_size = len(self.formation_space) if force_all else self.config['solidify_chunk_size']
        if chunk_size == 0: return
        chunk = [self.formation_space.popleft() for _ in range(chunk_size)]
        
        tangent_vec = np.mean(np.array([item['drip_head_state'] for item in chunk]), axis=0)
        new_point = np.array(PoincareBall.expmap0(jnp.array(tangent_vec)))
        self.H_sphere_points.append(new_point)
        
        # *** CRITICAL FIX: Store the start index and length of the chunk for later retrieval ***
        first_item = chunk[0]
        self.H_sphere_metadata.append({
            'start_token_idx': first_item['start_token_idx'],
            'chunk_len': len(chunk)
        })

    def save_cake(self, basename):
        print(f"--- Saving Funnel Cake to {basename}.cake ---")
        with open(f"{basename}.cake", 'wb') as f:
            pickle.dump({'config': self.config, 'ball_tree': self.ball_tree, 'H_sphere_metadata': self.H_sphere_metadata}, f)
        print("--- Cake saved. ---")
        
    def load_cake(self, basename):
        cake_file = f"{basename}.cake"
        print(f"--- Loading Funnel Cake from {cake_file} ---")
        if not os.path.exists(cake_file): print(f"[FATAL] Cake file not found: {cake_file}"), sys.exit(1)
        with open(cake_file, 'rb') as f: state = pickle.load(f)
        self.config, self.ball_tree, self.H_sphere_metadata = state['config'], state['ball_tree'], state['H_sphere_metadata']
        if self.ball_tree:
             print(f"--- Funnel Cake loaded. Contains {self.ball_tree.data.shape[0]:,} solidified points. ---")

    def generate(self, prompt, max_new=200, momentum=0.8):
        self.load_weights(self.config['basename'])
        self.load_cake(self.config['basename'])
        if self.nav_params is None or self.oracle_params is None or self.ball_tree is None:
            print("\n[ERROR] Navigator, Oracle, and Cake must be trained/constructed first.")
            return

        # --- Setup for Hierarchical Generation ---
        token_file_path = f"{self.config['basename']}.tokens.bin"
        if not os.path.exists(token_file_path):
            print(f"\n[ERROR] Token file '{token_file_path}' not found, which is required for hierarchical generation.")
            return
        tokens_memmap = np.memmap(token_file_path, dtype=np.int32, mode='r')
        print(f"\n\033[1;32m{prompt}\033[0m", end='', flush=True)

        cq_token_id = self.tokenizer.tokenizer.token_to_id("<CQ>")
        clarifying_token = cq_token_id if cq_token_id is not None else SamplerConfig.clarifying_question_token
        sampler_cfg = SamplerConfig(clarifying_question_token=clarifying_token)

        # --- JIT Compiled Helper Functions ---
        @jax.jit
        def get_navigator_state(params, tokens):
            h_r, h_i = self.navigator.apply({'params': params}, tokens)
            return h_r[:, -1, :], h_i[:, -1, :]

        @jax.jit
        def get_all_hidden_states(params, token_batch):
            h_r, h_i = self.navigator.apply({'params': params}, token_batch)
            return jnp.concatenate([h_r, h_i], axis=-1)

        @jax.jit
        def get_oracle_logits(params, h_complex):
            return self.oracle.apply({'params': params}, h_complex)
        
        # --- Initialization ---
        current_tokens = self.tokenizer.encode(prompt) or [self.tokenizer.tokenizer.token_to_id("<PAD>")]
        h_r, h_i = get_navigator_state(self.nav_params, jnp.array([current_tokens]))
        hidden_state_collapsed = jnp.concatenate([h_r.squeeze(), h_i.squeeze()])
        
        _, start_indices = self.ball_tree.query(np.array(hidden_state_collapsed, dtype=np.float32).reshape(1, -1), k=1)
        current_point = jnp.array(self.ball_tree.data[start_indices[0][0]])
        velocity_vector = jnp.zeros_like(current_point)
        
        # --- Main Generation Loop ---
        for _ in range(max_new):
            if self.should_shutdown: break
            
            self.key, subkey = jax.random.split(self.key)

            # === TIER 1: HIGH-LEVEL CONCEPT NAVIGATION (on the 256:1 Cake) ===
            current_anchor_state = hidden_state_collapsed
            intent_vector = PoincareBall.logmap0(current_point) - current_anchor_state
            
            k_for_flow = max(1, self.config['knn_sampling'])
            _, neighbor_indices_high_level = self.ball_tree.query(np.expand_dims(np.array(current_point), 0), k=k_for_flow)
            neighbors_high_level = jnp.array([self.ball_tree.data[idx] for idx in neighbor_indices_high_level.flatten()])
            local_flow_vector = jnp.mean(jax.vmap(PoincareBall.logmap0)(neighbors_high_level), axis=0)

            guidance_vector = (intent_vector * 0.8) + (local_flow_vector * 0.2)
            new_velocity = (velocity_vector * momentum) + (guidance_vector * (1 - momentum))
            velocity_vector = new_velocity / jnp.linalg.norm(new_velocity).clip(1e-6)
            
            # This is our target direction in the abstract concept space
            current_point = PoincareBall.expmap_p(current_point, velocity_vector * self.config['geodesic_step_size'])

            # === TIER 2: LOW-LEVEL TOKEN NAVIGATION (on a temporary 1:1 Micro-Cake) ===
            
            # 1. Identify the local neighborhood in the corpus
            k_for_micro_cake = self.config['micro_cake_neighbors']
            _, neighbor_indices = self.ball_tree.query(np.expand_dims(np.array(current_point), 0), k=k_for_micro_cake)
            
            # 2. Retrieve the raw tokens from these neighbors using our new metadata
            micro_cake_tokens = []
            for idx in neighbor_indices.flatten():
                meta = self.H_sphere_metadata[idx]
                start_idx, chunk_len = meta['start_token_idx'], meta['chunk_len']
                micro_cake_tokens.extend(tokens_memmap[start_idx : start_idx + chunk_len])
            
            if not micro_cake_tokens: continue # Safety check

            # 3. Build the temporary "Micro-Cake" on the fly
            micro_states = get_all_hidden_states(self.nav_params, jnp.array([micro_cake_tokens]))[0]
            micro_ball_tree = BallTree(np.array(micro_states, dtype=np.float32))

            # 4. Navigate within the micro-cake to find the best next state
            target_micro_state_tangent = current_anchor_state + velocity_vector * self.config['micro_step_size']
            _, best_next_idx = micro_ball_tree.query(np.array(target_micro_state_tangent).reshape(1, -1), k=1)
            final_oracle_input_state = micro_states[best_next_idx[0][0]]

            # === FINAL STEP: GENERATION & UPDATE ===
            
            # 1. Get logits from the Oracle using the highly-specific state found in the micro-cake
            oracle_input_h_r, oracle_input_h_i = jnp.split(final_oracle_input_state, 2)
            final_logits = get_oracle_logits(self.oracle_params, (oracle_input_h_r[None, None, :], oracle_input_h_i[None, None, :]))
            
            # 2. Sample using the XJDR Metacognitive Sampler
            next_token_id = xjdr_metacognitive_sample(subkey, final_logits.squeeze(), sampler_cfg).item()
            
            # 3. Decode and print
            decoded_token = self.tokenizer.tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(decoded_token.replace('Ġ', ' '), end='', flush=True)

            # 4. Update the running state for the next iteration
            current_tokens.append(next_token_id)
            if len(current_tokens) > 256: current_tokens.pop(0)

            h_r, h_i = get_navigator_state(self.nav_params, jnp.array([current_tokens]))
            hidden_state_collapsed = jnp.concatenate([h_r.squeeze(), h_i.squeeze()])
        print()


def main():
    parser = argparse.ArgumentParser(description="WubuMind Funnel Cake v24.1 (Hierarchical Oracle Edition)")
    parser.add_argument('command', choices=['pretokenize', 'train_navigator', 'train_oracle', 'construct', 'generate'], help="The command to execute.")
    parser.add_argument('--basename', type=str, default="wubumind_v24", help="Basename for model files.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=512, help="Batch size for training. Adjust based on VRAM.")
    parser.add_argument('--momentum', type=float, default=0.8, help="Momentum for the guidance system.")
    parser.add_argument('--max-new', type=int, default=200, help="Maximum new tokens to generate.")
    args = parser.parse_args()

    MODEL_CONFIG = { 
        'd_model': 256, 'solidify_chunk_size': 256, 'geodesic_step_size': 0.25, 
        'knn_sampling': 3, 'basename': args.basename, 'learning_rate': 1e-4, 'learning_rate_oracle': 5e-5,
        'train_chunk_size': 128,
        # --- NEW HIERARCHICAL CONFIG ---
        'micro_cake_neighbors': 5, # Number of high-level neighbors to build the micro-cake from
        'micro_step_size': 0.05,    # Smaller step size for fine-grained navigation
    }
    TOKENIZER_CONFIG = {'vocab_size': 8192, 'tokenizer_path': f"{args.basename}_bpe.json"}
    TOKEN_FILE_PATH = f"{args.basename}.tokens.bin"

    print(f"--- WubuMind Funnel Cake Foundry v24.1 (Hierarchical Oracle Edition) ---")
    
    tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
    
    if args.command == 'pretokenize':
        if not tokenizer.tokenizer:
            print("--- Consolidating CORPUS... ---")
            corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
            tokenizer.train((chunk for corpus in corpora for chunk in stream_text_from_corpus_data(corpus)), TOKENIZER_CONFIG['vocab_size'])

        print("--- Tokenizing corpus to binary file... ---")
        with open(TOKEN_FILE_PATH, 'wb') as f_out:
            corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
            for text_chunk in stream_text_from_corpus_data(corpora):
                if token_ids := tokenizer.encode(text_chunk): 
                    np.array(token_ids, dtype=np.int32).tofile(f_out)
        print("--- Pre-tokenization complete. ---")

    else:
        if not tokenizer.tokenizer:
            print(f"[FATAL] Tokenizer not found at '{TOKENIZER_CONFIG['tokenizer_path']}'. Run 'pretokenize' first.")
            sys.exit(1)

        constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)

        if args.command == 'train_navigator':
            if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file not found: {TOKEN_FILE_PATH}"), sys.exit(1)
            constructor.train_navigator(TOKEN_FILE_PATH, epochs=args.epochs, batch_size=args.batch_size)
        
        elif args.command == 'train_oracle':
            if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file not found: {TOKEN_FILE_PATH}"), sys.exit(1)
            constructor.train_oracle(TOKEN_FILE_PATH, epochs=args.epochs, batch_size=args.batch_size)

        elif args.command == 'construct':
            if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file not found: {TOKEN_FILE_PATH}"), sys.exit(1)
            constructor.construct(TOKEN_FILE_PATH, batch_size=args.batch_size)

        elif args.command == "generate":
            print("\n--- Hierarchical Oracle Command Console (v24.1) ---")
            while True:
                if constructor.should_shutdown: break
                try: prompt = input("\nYour Prompt> ")
                except EOFError: print("\n--- Exiting. ---"); break
                if prompt.lower() in ["exit", "quit"]: break
                constructor.generate(prompt, max_new=args.max_new, momentum=args.momentum)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
        sys.exit(0)
