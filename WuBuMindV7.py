import os
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import time
from tqdm import tqdm
import pickle
from typing import Any, Generator
import sys
import argparse
from collections import deque
import traceback
import faiss
import signal # <-- IMPORT FOR GRACEFUL EXIT

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

# --- Core Utilities ---
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
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<PAD>", "<UNK>"])
        self.tokenizer.train_from_iterator(corpus_iterator, trainer)
        self.tokenizer.save(self.tokenizer_path)
        print(f"--- Tokenizer trained. Vocab: {self.get_vocab_size()}. Saved to {self.tokenizer_path} ---")
    def get_vocab_size(self): return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    def encode(self, text): return self.tokenizer.encode(text).ids if self.tokenizer else []
    def decode(self, ids): return self.tokenizer.decode(ids) if self.tokenizer else ""

class PoincareBall:
    # NOTE: As per your instruction, hyperbolic math is done in float32 for stability,
    # then cast back to the original dtype.
    EPS = 1e-7
    @staticmethod
    def project(x):
        x_f32 = x.astype(jnp.float32); norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        max_norm = 1.0 - PoincareBall.EPS
        return jnp.where(norm_sq >= 1.0, x_f32 / jnp.sqrt(norm_sq).clip(PoincareBall.EPS) * max_norm, x_f32).astype(x.dtype)
    @staticmethod
    def mobius_add(x, y, c=1.0):
        x_f32, y_f32 = x.astype(jnp.float32), y.astype(jnp.float32)
        x2, y2, xy = jnp.sum(x_f32*x_f32, -1, keepdims=True), jnp.sum(y_f32*y_f32, -1, keepdims=True), jnp.sum(x_f32*y_f32, -1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x_f32 + (1 - c * x2) * y_f32
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS)).astype(x.dtype)
    @staticmethod
    def logmap0(y, c=1.0):
        y_f32 = y.astype(jnp.float32); sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True); safe_norm = y_norm.clip(PoincareBall.EPS)
        return jnp.where(safe_norm > 0, jnp.arctanh(y_norm.clip(max=1.0-PoincareBall.EPS)) * y_f32 / (sqrt_c * safe_norm), jnp.zeros_like(y_f32))
    @staticmethod
    def expmap0(v, c=1.0):
        v_f32 = v.astype(jnp.float32); sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True); safe_norm = v_norm.clip(PoincareBall.EPS)
        return jnp.where(safe_norm > 0, PoincareBall.project(jnp.tanh(sqrt_c * safe_norm) * v_f32 / (sqrt_c * safe_norm)), jnp.zeros_like(v_f32))
    @staticmethod
    def expmap_p(p, v, c=1.0):
        p_f32, v_f32 = p.astype(jnp.float32), v.astype(jnp.float32)
        lambda_p = 2. / (1 - c * jnp.sum(p_f32*p_f32, axis=-1, keepdims=True)).clip(PoincareBall.EPS)
        return PoincareBall.mobius_add(p_f32, PoincareBall.expmap0(v_f32 * lambda_p, c), c)

# --- Drip Head (Batched Architecture for Speed) ---
class GRUScanBody(nn.Module):
    d_model: int; dtype: Any
    @nn.compact
    def __call__(self, carry, x_token_embed):
        gru_cell = nn.GRUCell(features=self.d_model, name="gru_cell", dtype=self.dtype)
        new_carry, _ = gru_cell(carry, x_token_embed)
        return new_carry, new_carry

class DripHead(nn.Module):
    d_model: int; vocab_size: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, token_ids, initial_carry):
        token_embeds = nn.Embed(self.vocab_size, self.d_model, name="token_embed", dtype=self.dtype)(token_ids)
        scanner = nn.scan(GRUScanBody, variable_broadcast='params', split_rngs={'params': False}, in_axes=0, out_axes=0)
        final_carry, hidden_states = scanner(d_model=self.d_model, dtype=self.dtype)(initial_carry, token_embeds)
        return final_carry, hidden_states

# --- Funnel Cake Constructor ---
class FunnelCakeConstructor:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer; self.d_model = config['d_model']
        self.key = jax.random.PRNGKey(42)
        self.drip_head = DripHead(d_model=self.d_model, vocab_size=tokenizer.get_vocab_size())
        self.drip_head_params = None; self.formation_space = deque()
        self.H_sphere_points = np.empty((0, self.d_model), dtype=np.float32)
        self.H_sphere_metadata = []; self.hyperbolic_index = None
        
        # --- GRACEFUL EXIT: Setup signal handler ---
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if not self.should_shutdown:
            print("\n--- SIGINT received. Will save and exit after the current batch. Press again to force quit. ---")
            self.should_shutdown = True
        else:
            print("\n--- Forced exit. ---")
            sys.exit(1)

    def _init_drip_head(self):
        print("--- Initializing Drip Head (Random Projection Mode)... ---")
        self.key, drip_key = jax.random.split(self.key)
        dummy_tokens = jnp.zeros((16,), dtype=jnp.int32)
        dummy_hidden = jnp.zeros((self.d_model,), dtype=self.drip_head.dtype)
        self.drip_head_params = self.drip_head.init(drip_key, dummy_tokens, dummy_hidden)['params']
        param_count = sum(x.size for x in jax.tree.leaves(self.drip_head_params))
        print(f"--- Drip Head Initialized: {param_count:,} params. No training will be performed. ---")

    def construct(self, corpus_filepath):
        if self.drip_head_params is None: self._init_drip_head()
        print(f"--- Constructing Funnel Cake from '{corpus_filepath}'... ---")
        
        batch_size = 256
        @jax.jit
        def drip_batch_step(params, token_batch, initial_hidden):
            final_hidden, all_hidden = self.drip_head.apply({'params': params}, token_batch, initial_hidden)
            return final_hidden, all_hidden

        hidden_state = jnp.zeros((self.d_model,), dtype=self.drip_head.dtype)
        
        file_size = os.path.getsize(corpus_filepath)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Constructing Filament")
        
        last_update_time = time.time(); tokens_since_last_update = 0

        with open(corpus_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            while (text_chunk := f.read(128 * 1024)):
                token_ids = self.tokenizer.encode(text_chunk)
                
                for i in range(0, len(token_ids), batch_size):
                    batch = jnp.array(token_ids[i:i+batch_size], dtype=jnp.int32)
                    if len(batch) == 0: continue
                    
                    final_hidden, batch_hidden_states = drip_batch_step(self.drip_head_params, batch, hidden_state)
                    hidden_state = final_hidden
                    
                    # --- PERFORMANCE FIX: Pull data from device to CPU in one go ---
                    batch_cpu = np.array(batch)
                    hidden_states_cpu = np.array(batch_hidden_states)

                    for j in range(len(batch_cpu)):
                        self.formation_space.append({
                            'drip_head_state': hidden_states_cpu[j],
                            'token_id': batch_cpu[j].item() # Now a fast numpy op
                        })
                    
                    while len(self.formation_space) >= self.config['solidify_chunk_size']:
                        self._solidify()
                    
                    tokens_since_last_update += len(batch_cpu)

                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed > 1.0:
                    rate_tokens_s = tokens_since_last_update / elapsed
                    pbar.set_postfix(solids=f"{len(self.H_sphere_metadata)}", rate=f"{rate_tokens_s:,.0f} tok/s")
                    last_update_time = current_time; tokens_since_last_update = 0

                pbar.update(len(text_chunk.encode('utf-8')))

                # --- GRACEFUL EXIT: Check the flag after processing a text chunk ---
                if self.should_shutdown:
                    print("\n--- Interrupt honored. Finishing up... ---")
                    break
        
        if self.formation_space: self._solidify(force_all=True)
        pbar.close()
        print(f"\n--- Construction Complete. Solids: {self.H_sphere_points.shape[0]} ---")
        self.save(self.config['basename'])

    # ... The rest of the class (_solidify, save, load, generate) is unchanged ...
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
        state = {'config': self.config, 'drip_head_params': jax.device_get(self.drip_head_params),
                 'H_sphere_points': self.H_sphere_points, 'H_sphere_metadata': self.H_sphere_metadata}
        with open(f"{basename}.cake", 'wb') as f: pickle.dump(state, f)
        print("--- Save complete. ---")

    def load(self, basename):
        print(f"--- Loading Funnel Cake from {basename}.cake ---")
        with open(f"{basename}.cake", 'rb') as f: state = pickle.load(f)
        self.config = state['config']
        self._init_drip_head()
        self.drip_head_params = state['drip_head_params']
        self.H_sphere_points = state['H_sphere_points']
        self.H_sphere_metadata = state['H_sphere_metadata']
        if self.H_sphere_points.shape[0] > 0:
            self.hyperbolic_index = faiss.IndexFlatL2(self.d_model)
            self.hyperbolic_index.add(self.H_sphere_points.astype(np.float32))
        print(f"--- Funnel Cake loaded. Contains {self.H_sphere_points.shape[0]} solidified points. ---")

    def generate(self, prompt, max_new=200, momentum=0.8):
        if self.hyperbolic_index is None: print("\n[ERROR] Funnel Cake is empty."), sys.exit(1)
        print(f"\n\033[1;32m{prompt}\033[0m", end=''); sys.stdout.flush()
        
        hidden_state = jnp.zeros((self.d_model,), dtype=self.drip_head.dtype)
        prompt_tokens = self.tokenizer.encode(prompt)
        if prompt_tokens:
             final_hidden, _ = self.drip_head.apply({'params': self.drip_head_params}, jnp.array(prompt_tokens), hidden_state)
             hidden_state = final_hidden

        _, start_indices = self.hyperbolic_index.search(np.array(hidden_state, dtype=np.float32).reshape(1, -1), 1)
        current_point = jnp.array(self.H_sphere_points[start_indices[0][0]])
        
        velocity_vector = PoincareBall.logmap0(current_point) - hidden_state
        velocity_vector /= jnp.linalg.norm(velocity_vector).clip(1e-6)

        for _ in range(max_new):
            _, neighbor_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), self.config['knn_sampling'])
            neighbors = jnp.array(self.H_sphere_points[neighbor_indices.flatten()])
            tangent_vectors = jax.vmap(PoincareBall.logmap0)(neighbors)
            local_flow_vector = jnp.mean(tangent_vectors, axis=0)
            local_flow_direction = local_flow_vector / jnp.linalg.norm(local_flow_vector).clip(1e-6)
            
            new_velocity = (velocity_vector * momentum) + (local_flow_direction * (1 - momentum))
            velocity_vector = new_velocity / jnp.linalg.norm(new_velocity).clip(1e-6)

            current_point = PoincareBall.expmap_p(current_point, velocity_vector * self.config['geodesic_step_size'])
            
            _, closest_indices = self.hyperbolic_index.search(np.expand_dims(np.array(current_point), 0), 1)
            chosen_neighbor_idx = closest_indices.flatten()[0]
            retrieved_chunk_tokens = self.H_sphere_metadata[chosen_neighbor_idx]['token_ids']
            if not retrieved_chunk_tokens: continue
            next_token_id = np.random.choice(retrieved_chunk_tokens)
            
            print(self.tokenizer.decode([next_token_id]), end=''); sys.stdout.flush()
            
            final_hidden, _ = self.drip_head.apply({'params': self.drip_head_params}, jnp.array([next_token_id]), hidden_state)
            hidden_state = final_hidden
        print()


def main():
    parser = argparse.ArgumentParser(description="WubuMind Funnel Cake Constructor v6.3 (Graceful Exit & Performance)")
    parser.add_argument('command', choices=['construct', 'generate'], help="The command to execute.")
    parser.add_argument('--basename', type=str, default="wubumind_funnel_cake_v1", help="Basename for model files.")
    args = parser.parse_args()

    MODEL_CONFIG = {'d_model': 128, 'solidify_chunk_size': 256, 'knn': 5, 'geodesic_step_size': 0.05, 'knn_sampling': 3, 'basename': args.basename}
    TOKENIZER_CONFIG = {'vocab_size': 4096, 'tokenizer_path': f"{args.basename}_bpe.json"}
    CORPUS_FILE_PATH = f"{args.basename}.corpus.txt"

    print(f"--- WubuMind Funnel Cake Foundry v6.3 (Graceful Exit & Performance) ---")
    corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
    if not corpora: print("[FATAL] No CORPUS vars found in CORPUS.py."), sys.exit(1)
    
    if not os.path.exists(CORPUS_FILE_PATH):
        print(f"Consolidating CORPUS into a single file: '{CORPUS_FILE_PATH}'...")
        with open(CORPUS_FILE_PATH, 'w', encoding='utf-8') as f:
            for text_chunk in stream_text_from_corpus_data(corpora): f.write(text_chunk + "\n")
        print("Corpus consolidation complete.")

    tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
    if not tokenizer.tokenizer:
        tokenizer.train((line for line in open(CORPUS_FILE_PATH, 'r', encoding='utf-8')), TOKENIZER_CONFIG['vocab_size'])
        print("\n--- Tokenizer trained. Please run 'construct' again. ---"), sys.exit(0)

    constructor = FunnelCakeConstructor(MODEL_CONFIG, tokenizer)

    if args.command == "construct":
        constructor.construct(CORPUS_FILE_PATH)
    
    elif args.command == "generate":
        try:
            constructor.load(args.basename)
        except FileNotFoundError:
            print(f"[FATAL] No model found at '{args.basename}.cake'. Please run 'construct' first."), sys.exit(1)

        print("\n--- Oracle Command Console (Funnel Cake Edition) ---")
        while True:
            try:
                prompt = input("\nYour Prompt> ")
                if prompt.lower() in ["exit", "quit"]: break
                constructor.generate(prompt)
            except KeyboardInterrupt: print("\n-- Exiting. --"); break
            except Exception as e: print(f"\nAn error occurred: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()