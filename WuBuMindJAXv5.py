# %% PYTHON FILE: wubumind_galactic_core_v1.py
# The Phoenix. Final, robust version with live streaming inference and corrected training init.

import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import remat
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import numpy as np
import time
from tqdm import tqdm
import pickle
import json
from typing import Any, Sequence, Dict, Tuple
import sys
import dataclasses
import signal
import traceback

# --- CORPUS IMPORT ---
try:
    import CORPUS
except ImportError:
    print("[FATAL ERROR] CORPUS.py not found. Please create it and add the CORPUS data.")
    sys.exit(1)

# Ensure full GPU memory is available for JAX.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_debug_nans", False)

# --- Helper Functions & Converters (Unchanged) ---
def distill_text_from_corpus(data: Any) -> str:
    if isinstance(data, str): return data + "\n"
    elif isinstance(data, dict): return "".join(distill_text_from_corpus(v) for v in data.values())
    elif isinstance(data, list): return "".join(distill_text_from_corpus(item) for item in data)
    return ""

def create_corpus_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    print(f"--- Discovering absolute vocabulary from CORPUS... ---")
    unique_chars = sorted(list(set(text)))
    idx_to_char = {0: '<PAD>', 1: '<UNK>'}
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, char in enumerate(unique_chars):
        idx = i + 2
        idx_to_char[idx] = char
        char_to_idx[char] = idx
    print(f"--- Absolute vocabulary created. Size: {len(idx_to_char)} ---")
    return char_to_idx, idx_to_char

class UnicodeGeometrodynamicConverter:
    def __init__(self, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str]):
        self.char_to_idx, self.idx_to_char = char_to_idx, idx_to_char
        self.vocab_size, self.unk_idx, self.pad_idx = len(char_to_idx), char_to_idx.get('<UNK>', 1), char_to_idx.get('<PAD>', 0)
    def get_indices_from_text(self, text:str) -> list[int]:
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]

class RollingHasher:
    def __init__(self, window_size, base=313, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus, self.precomputed_base = window_size, base, modulus, pow(base, window_size - 1, modulus)
    def hash_sequence(self, values: np.ndarray):
        num_values = len(values)
        if num_values < self.window_size: return np.array([], dtype=np.int32)
        hashes = np.zeros(num_values - self.window_size + 1, dtype=np.int32)
        current_hash = 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + int(values[i])) % self.modulus
        hashes[0] = current_hash
        for i in range(1, num_values - self.window_size + 1):
            current_hash = ((current_hash - int(values[i-1]) * self.precomputed_base) * self.base + int(values[i+self.window_size-1])) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes[i] = current_hash
        return hashes

# --- Geometric Primitives (Unchanged) ---
class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def project(x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(PoincareBall.EPS)
        max_norm = 1.0 - PoincareBall.EPS; cond = norm >= 1.0
        return jnp.where(cond, x / norm * max_norm, x)
    @staticmethod
    def mobius_add(x, y, c):
        x2,y2,xy = jnp.sum(x*x,-1,keepdims=True),jnp.sum(y*y,-1,keepdims=True),jnp.sum(x*y,-1,keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS))
    @staticmethod
    def logmap0(y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS); y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        safe_y_norm = y_norm.clip(PoincareBall.EPS); direction = y / safe_y_norm
        magnitude = jnp.arctanh(y_norm.clip(max=1.0-PoincareBall.EPS)) / sqrt_c
        return jnp.where(y_norm < PoincareBall.EPS, jnp.zeros_like(y), magnitude * direction)
    @staticmethod
    def expmap0(v, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS); v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        safe_v_norm = v_norm.clip(PoincareBall.EPS); direction = v / safe_v_norm
        magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c
        return PoincareBall.project(jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v), magnitude * direction))
    @staticmethod
    def dist(x, y, c):
        c_bcast = c[..., None]
        sqrt_c = jnp.sqrt(c_bcast).clip(PoincareBall.EPS)
        add_xy = PoincareBall.mobius_add(-x, y, c_bcast)
        add_norm = jnp.linalg.norm(add_xy, axis=-1)
        arg = jnp.minimum(sqrt_c.squeeze(-1) * add_norm, 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg) / sqrt_c.squeeze(-1)

# --- GALACTIC CORE ARCHITECTURE (Unchanged) ---
class HyperbolicAttention(nn.Module):
    dim: int; n_heads: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        x_f32 = x.astype(jnp.float32)
        x_r, x_i = jnp.split(x_f32, 2, -1)
        x_c = jax.lax.complex(x_r, x_i)
        freqs_cis = freqs_cis.reshape(1, 1, freqs_cis.shape[0], freqs_cis.shape[1])
        x_rotated = x_c * freqs_cis
        return jnp.concatenate([x_rotated.real, x_rotated.imag], -1).astype(x.dtype)
    @nn.compact
    def __call__(self, x_hyp, freqs_cis, c_sphere):
        B, N, _ = x_hyp.shape
        h_dim = self.dim // self.n_heads
        qkv_proj = nn.Dense(self.dim * 3, name="qkv_proj", dtype=self.dtype, param_dtype=self.param_dtype)
        out_proj = nn.Dense(self.dim, name="out_proj", dtype=self.dtype, param_dtype=self.param_dtype)
        c_per_head_logits = self.param('c_per_head_logits', nn.initializers.zeros, (self.n_heads,), self.param_dtype)
        geo_scale = self.param('geo_scale', nn.initializers.ones, (1, self.n_heads, 1, 1), self.param_dtype)
        x_tangent = PoincareBall.logmap0(x_hyp, c_sphere)
        qkv = qkv_proj(x_tangent).reshape(B, N, 3, self.n_heads, h_dim).transpose((2, 0, 3, 1, 4))
        q, k, v_euc = qkv[0], qkv[1], qkv[2]
        q_rot, k_rot = self.apply_rotary_emb(q, freqs_cis), self.apply_rotary_emb(k, freqs_cis)
        c_per_head = nn.softplus(c_per_head_logits).reshape(1, self.n_heads, 1, 1)
        q_hyp, k_hyp = PoincareBall.expmap0(q_rot, c_per_head), PoincareBall.expmap0(k_rot, c_per_head)
        q_bcast, k_bcast = q_hyp[:, :, :, None, :], k_hyp[:, :, None, :, :]
        dist = PoincareBall.dist(q_bcast, k_bcast, c_per_head)
        mask = nn.make_causal_mask(jnp.ones((B, N), dtype=bool))
        attn_scores = jnp.where(mask, -geo_scale * dist, -jnp.inf)
        attn_weights = nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        attn_out_euc = (attn_weights @ v_euc).transpose((0, 2, 1, 3)).reshape(B, N, self.dim)
        return out_proj(attn_out_euc)

class HyperbolicFFN(nn.Module):
    dim: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x_hyp, c_sphere):
        x_tangent = PoincareBall.logmap0(x_hyp, c_sphere)
        return nn.Sequential([nn.Dense(self.dim*4,dtype=self.dtype,param_dtype=self.param_dtype), nn.gelu, nn.Dense(self.dim,dtype=self.dtype,param_dtype=self.param_dtype)])(x_tangent)

class GalacticBlock(nn.Module):
    dim: int; n_heads: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @remat
    @nn.compact
    def __call__(self, states: Dict[str, jnp.ndarray], curvatures: Dict[str, jnp.ndarray], freqs_cis: jnp.ndarray):
        x_euc, x_syn, x_sem, x_exe = states['euc'], states['syn'], states['sem'], states['exe']
        c_syn, c_sem, c_exe = curvatures['syn'], curvatures['sem'], curvatures['exe']
        norm_euc_1 = nn.LayerNorm(dtype=jnp.float32, name="norm_euc_1")(x_euc).astype(self.dtype)
        syn_informed = PoincareBall.mobius_add(x_syn, PoincareBall.expmap0(norm_euc_1, c_syn), c_syn)
        sem_informed = PoincareBall.mobius_add(x_sem, PoincareBall.expmap0(norm_euc_1, c_sem), c_sem)
        exe_informed = PoincareBall.mobius_add(x_exe, PoincareBall.expmap0(norm_euc_1, c_exe), c_exe)
        attn_update_syn = HyperbolicAttention(self.dim, self.n_heads, name="attn_syn", dtype=self.dtype, param_dtype=self.param_dtype)(syn_informed, freqs_cis, c_syn)
        attn_update_sem = HyperbolicAttention(self.dim, self.n_heads, name="attn_sem", dtype=self.dtype, param_dtype=self.param_dtype)(sem_informed, freqs_cis, c_sem)
        attn_update_exe = HyperbolicAttention(self.dim, self.n_heads, name="attn_exe", dtype=self.dtype, param_dtype=self.param_dtype)(exe_informed, freqs_cis, c_exe)
        chassis_attn_update = attn_update_syn + attn_update_sem + attn_update_exe
        x_euc_post_attn = x_euc + chassis_attn_update
        norm_euc_2 = nn.LayerNorm(dtype=jnp.float32, name="norm_euc_2")(x_euc_post_attn).astype(self.dtype)
        ffn_update_syn = HyperbolicFFN(self.dim, name="ffn_syn", dtype=self.dtype, param_dtype=self.param_dtype)(x_syn, c_syn)
        ffn_update_sem = HyperbolicFFN(self.dim, name="ffn_sem", dtype=self.dtype, param_dtype=self.param_dtype)(x_sem, c_sem)
        ffn_update_exe = HyperbolicFFN(self.dim, name="ffn_exe", dtype=self.dtype, param_dtype=self.param_dtype)(x_exe, c_exe)
        comm_sem_to_syn = nn.Dense(self.dim, name="comm_sem_to_syn", dtype=self.dtype, param_dtype=self.param_dtype)(ffn_update_sem)
        comm_exe_to_sem = nn.Dense(self.dim, name="comm_exe_to_sem", dtype=self.dtype, param_dtype=self.param_dtype)(ffn_update_exe)
        x_syn_final = PoincareBall.expmap0(PoincareBall.logmap0(x_syn, c_syn) + ffn_update_syn + comm_sem_to_syn, c_syn)
        x_sem_final = PoincareBall.expmap0(PoincareBall.logmap0(x_sem, c_sem) + ffn_update_sem + comm_exe_to_sem, c_sem)
        x_exe_final = PoincareBall.expmap0(PoincareBall.logmap0(x_exe, c_exe) + ffn_update_exe, c_exe)
        chassis_ffn_update = ffn_update_syn + ffn_update_sem + ffn_update_exe
        gate = nn.Dense(self.dim, name="chassis_gate", kernel_init=nn.initializers.zeros, dtype=self.dtype, param_dtype=self.param_dtype)(norm_euc_2)
        x_euc_final = x_euc_post_attn + chassis_ffn_update * nn.sigmoid(gate)
        return {'euc': x_euc_final, 'syn': x_syn_final, 'sem': x_sem_final, 'exe': x_exe_final}

@dataclasses.dataclass
class WubuMind(nn.Module):
    vocab_size: int; d_model: int; n_heads: int; n_layers: int; modulus: int
    hash_window: int; max_len: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
        freqs = 1.0/(theta**(jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        return jnp.exp(1j * jnp.outer(jnp.arange(end), freqs))
    @nn.compact
    def __call__(self, indices, hashes):
        B, N = indices.shape
        h_dim = self.d_model // self.n_heads
        token_embed = nn.Embed(self.vocab_size, self.d_model // 2, dtype=self.dtype, param_dtype=self.param_dtype, name="token_embed")(indices)
        hash_embed = nn.Dense(self.d_model // 2, dtype=self.dtype, param_dtype=self.param_dtype, name="hash_proj")((hashes[...,None] / self.modulus).astype(self.dtype))
        base_euc = nn.Dense(self.d_model, name="bridge_proj", dtype=self.dtype, param_dtype=self.param_dtype)(jnp.concatenate([token_embed, hash_embed], axis=-1))
        c_syn, c_sem, c_exe = nn.softplus(self.param('c_syntactic', nn.initializers.constant(5.0),(1,))), nn.softplus(self.param('c_semantic', nn.initializers.constant(1.0),(1,))), nn.softplus(self.param('c_executive', nn.initializers.constant(0.1),(1,)))
        x_syn = PoincareBall.expmap0(nn.Dense(self.d_model, name="proj_syntactic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc), c_syn)
        x_sem = PoincareBall.expmap0(nn.Dense(self.d_model, name="proj_semantic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc), c_sem)
        x_exe = PoincareBall.expmap0(nn.Dense(self.d_model, name="proj_executive",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc), c_exe)
        x_euc_chassis = nn.LayerNorm(dtype=self.dtype, name="proj_chassis_norm")(base_euc)
        states = {'euc': x_euc_chassis, 'syn': x_syn, 'sem': x_sem, 'exe': x_exe}
        curvatures = {'syn': c_syn, 'sem': c_sem, 'exe': c_exe}
        freqs_cis = self.precompute_freqs_cis(h_dim, self.max_len)[:N]
        for i in range(self.n_layers):
            states = GalacticBlock(dim=self.d_model, n_heads=self.n_heads, name=f"galaxy_{i}", dtype=self.dtype, param_dtype=self.param_dtype)(states, curvatures, freqs_cis)
        final_x_euc = nn.LayerNorm(dtype=jnp.float32, name="final_norm")(states['euc'])
        return nn.Dense(self.vocab_size, dtype=jnp.float32, name="output_proj")(final_x_euc)

# --- DATA PREPARATION, GRADIENT STEP, AND CHECKPOINTING (Unchanged) ---
def prepare_training_data_on_device(text_corpus, converter, hasher, config):
    print("--- Beginning high-performance data pipeline... ---")
    indices = np.array(converter.get_indices_from_text(text_corpus), dtype=np.int32)
    context_length, batch_size = config['context_length'], config['batch_size']
    hashes = hasher.hash_sequence(indices)
    num_samples = min(len(indices) - context_length - 1, len(hashes) - context_length)
    strides_i, strides_h = indices.strides[0], hashes.strides[0]
    all_indices = np.lib.stride_tricks.as_strided(indices, shape=(num_samples, context_length), strides=(strides_i, strides_i))
    all_targets = np.lib.stride_tricks.as_strided(indices[1:], shape=(num_samples, context_length), strides=(strides_i, strides_i))
    all_hashes = np.lib.stride_tricks.as_strided(hashes, shape=(num_samples, context_length), strides=(strides_h, strides_h))
    num_batches = num_samples // batch_size
    if num_to_trim := num_samples % batch_size:
        all_indices, all_hashes, all_targets = [arr[:-num_to_trim] for arr in (all_indices, all_hashes, all_targets)]
    all_indices_batched, all_hashes_batched, all_targets_batched = [arr.reshape(num_batches, batch_size, context_length) for arr in (all_indices, all_hashes, all_targets)]
    print(f"--- Data preparation complete. {num_batches} micro-batches created. ---")
    print("--- Transferring all batches to device... ---")
    device_batches = jax.device_put((all_indices_batched, all_hashes_batched, all_targets_batched))
    print("--- Data pipeline complete. All data is now on-device. ---")
    return device_batches, num_batches

@partial(jax.jit, static_argnames=['apply_fn'])
def grad_computation_step(params, apply_fn, batch):
    indices, hashes, targets = batch
    def loss_fn(p):
        logits = apply_fn({'params': p}, indices, hashes)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

def save_checkpoint(state, basename):
    with open(f"{basename}.pkl", 'wb') as f:
        pickle.dump(serialization.to_state_dict(jax.device_get(state)), f)
    print(f"\n--- Checkpoint saved. Step: {state.step} ---")

def load_checkpoint(state, basename):
    filename = f"{basename}.pkl"
    if not os.path.exists(filename):
        print("--- No checkpoint found. Starting from scratch. JIT Compilation Will Mean Delay on first run.  ---")
        return state
    with open(filename, 'rb') as f: saved_state_dict = pickle.load(f)
    print(f"--- Checkpoint found. Restoring full state... ---")
    restored_state = serialization.from_state_dict(state, saved_state_dict)
    print(f"--- Full training state restored. JIT Compilation Will Mean Delay on first run. Resuming from step: {restored_state.step} ---")
    return restored_state

def save_config(config, char_to_idx, basename):
    serializable_config = {k: v for k, v in config.items() if k != 'char_to_idx'}
    with open(f"{basename}.json", 'w') as f: json.dump(serializable_config, f, indent=4)
    with open(f"{basename}_vocab.json", 'w', encoding='utf-8') as f: json.dump(char_to_idx, f, indent=4)
    print(f"--- Model config and vocab saved to {basename}.json / _vocab.json ---")

# --- THE PHOENIX: ROBUST TRAINING MANAGER (Unchanged) ---
class TrainingManager:
    def __init__(self, model, config, data):
        self.model, self.config, self.data = model, config, data
        self.basename = "wubumind_galactic_core_v1"
        self.state, self.should_shutdown = None, False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if not self.should_shutdown:
            print("\n--- SIGINT received. Saving state after current step... ---")
            self.should_shutdown = True

    def run(self):
        (i_all, h_all, t_all), micro_batches_total = self.data
        ga_steps = self.config['gradient_accumulation_steps']
        micro_batches_per_epoch = micro_batches_total // self.config['epochs']
        if micro_batches_per_epoch == 0:
            print(f"[FATAL ERROR] Not enough data for one epoch. Need at least {self.config['epochs'] * ga_steps} samples, but found {micro_batches_total}.")
            return

        key = jax.random.PRNGKey(42)
        params = self.model.init(jax.random.split(key)[1], i_all[0][:1], h_all[0][:1])['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f'--- Model initialized with {param_count:,} parameters. ---')
        
        total_steps = self.config['epochs'] * (micro_batches_per_epoch // ga_steps)
        lr_schedule = optax.warmup_cosine_decay_schedule(0.0, self.config['peak_learning_rate'], self.config['warmup_steps'], total_steps, end_value=self.config['peak_learning_rate']/10)
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
        
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        save_config(self.config, self.config['char_to_idx'], self.basename)
        self.state = load_checkpoint(self.state, self.basename)

        start_step = self.state.step
        if start_step >= total_steps:
            print(f"--- Training already completed at step {start_step}. To train further, increase 'epochs'. ---")
            return

        grad_accumulator, loss_accumulator = jax.tree_util.tree_map(jnp.zeros_like, self.state.params), 0.0

        with tqdm(total=total_steps, initial=start_step, desc="Training", unit="step") as pbar:
            for step in range(start_step, total_steps):
                pbar.set_description(f"Epoch {(step // (micro_batches_per_epoch // ga_steps)) + 1}/{self.config['epochs']}")
                for micro_step in range(ga_steps):
                    batch_idx = (step * ga_steps + micro_step) % micro_batches_total
                    batch = (i_all[batch_idx], h_all[batch_idx], t_all[batch_idx])
                    loss, grads = grad_computation_step(self.state.params, self.state.apply_fn, batch)
                    if not jnp.isnan(loss):
                        grad_accumulator = jax.tree_util.tree_map(lambda acc, g: acc + g, grad_accumulator, grads)
                        loss_accumulator += loss
                
                self.state = self.state.apply_gradients(grads=jax.tree_util.tree_map(lambda g: g / ga_steps, grad_accumulator))
                pbar.set_postfix(loss=f"{(loss_accumulator / ga_steps):.4f}", lr=f"{lr_schedule(self.state.step):.2e}")
                pbar.update(1)
                grad_accumulator, loss_accumulator = jax.tree_util.tree_map(jnp.zeros_like, self.state.params), 0.0

                if self.should_shutdown:
                    print("\n--- Interrupt signal honored. Saving final state. ---")
                    save_checkpoint(self.state, self.basename)
                    return
        
        print("\n--- Training loop finished normally. ---")
        save_checkpoint(self.state, self.basename)

# --- TRAINING MAIN ---
def training_main():
    MODEL_CONFIG_BASE = {'hash_window': 16, 'd_model': 384, 'n_heads': 6, 'n_layers': 8, 'modulus': 10**9 + 7, 'max_len': 4096}
    TRAINING_CONFIG = {'epochs': 20, 'batch_size': 2, 'gradient_accumulation_steps': 8, 'peak_learning_rate': 5e-4, 'warmup_steps': 500, 'context_length': 256}
    
    print(f"--- WubuMind Galactic Core Foundry ---\n--- Using device: {jax.devices()[0].platform.upper()} ---")
    
    print("--- Automatically discovering CORPUS data from CORPUS.py... ---")
    discovered_corpora = [getattr(CORPUS, name) for name in dir(CORPUS) if not name.startswith('_') and name.isupper()]
    if not discovered_corpora: print("[FATAL ERROR] No CORPUS variables in CORPUS.py."), sys.exit(1)

    corpus_text = distill_text_from_corpus(discovered_corpora)
    try: corpus_text += open(__file__, 'r', encoding='utf-8').read()
    except Exception: print("Could not self-read.")
    
    print(f"--- Total characters in CORPUS: {len(corpus_text):,} ---")

    char_to_idx, idx_to_char = create_corpus_vocab(corpus_text)
    converter = UnicodeGeometrodynamicConverter(char_to_idx, idx_to_char)
    
    # --- THIS IS THE CORRECTED LINE ---
    arch_config = {**MODEL_CONFIG_BASE, 'vocab_size': len(char_to_idx)}
    # ------------------------------------

    FULL_CONFIG = {**arch_config, **TRAINING_CONFIG, 'char_to_idx': char_to_idx}
    hasher = RollingHasher(arch_config['hash_window'])
    data_bundle = prepare_training_data_on_device(corpus_text, converter, hasher, FULL_CONFIG)
    model = WubuMind(**arch_config)
    TrainingManager(model, FULL_CONFIG, data_bundle).run()

# --- INFERENCE AND MAIN ---
@partial(jax.jit, static_argnames=['model_apply_fn', 'temp', 'top_p'])
def predict_step_fn(model_apply_fn, params, indices, hashes, key, temp, top_p):
    logits = model_apply_fn({'params': params}, indices, hashes)
    scaled = logits[:, -1, :] / jnp.maximum(temp, 1e-6)
    if top_p < 1.0:
        sorted_indices = jnp.argsort(scaled, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(scaled, sorted_indices, axis=-1)
        cum_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_to_remove = cum_probs > top_p
        sorted_to_remove = jnp.concatenate([jnp.zeros_like(sorted_to_remove[..., :1]), sorted_to_remove[..., :-1]], axis=-1)
        to_remove = jnp.zeros_like(sorted_to_remove).at[..., sorted_indices].set(sorted_to_remove)
        scaled = jnp.where(to_remove, -jnp.inf, scaled)
    return jax.random.categorical(key, scaled, axis=-1)

class WubuOracle:
    def __init__(self, model_basename: str):
        print("--- The Oracle is Awakening ---")
        self.basename = model_basename
        with open(f"{self.basename}.json", 'r') as f: self.config = json.load(f)
        with open(f"{self.basename}_vocab.json", 'r', encoding='utf-8') as f: char_to_idx = json.load(f)
        idx_to_char = {int(v): k for k, v in char_to_idx.items()}
        self.converter = UnicodeGeometrodynamicConverter(char_to_idx, idx_to_char)
        self.hasher = RollingHasher(self.config['hash_window'])
        model_fields = [f.name for f in dataclasses.fields(WubuMind)]
        arch_config = {k: v for k, v in self.config.items() if k in model_fields}
        self.model = WubuMind(**arch_config)
        self.jit_compiled = False
        print("--- Assimilating knowledge from checkpoint... ---")
        
        dummy_lr_schedule = optax.warmup_cosine_decay_schedule(0.0, 1.0, 1, 2)
        dummy_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(dummy_lr_schedule, weight_decay=0.01))
        
        dummy_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(jax.random.PRNGKey(0), jnp.ones((1,1),dtype=jnp.int32), jnp.ones((1,1),dtype=jnp.int32))['params'],
            tx=dummy_tx
        )
        
        with open(f"{self.basename}.pkl", 'rb') as f: saved_state_dict = pickle.load(f)
        state = serialization.from_state_dict(dummy_state, saved_state_dict)
        self.params = state.params
        self.predict_step = partial(predict_step_fn, self.model.apply)
        print(f"--- Oracle has assimilated knowledge from step {state.step}. Ready to Speak. ---")

    def generate(self, prompt: str, max_new: int = 500, temp: float = 0.7, top_p: float = 0.95):
        if not self.jit_compiled:
            print("--- JIT compiling model for first use... (this may take a moment) ---", flush=True)

        key = jax.random.PRNGKey(int(time.time()))
        indices = self.converter.get_indices_from_text(prompt)
        
        sys.stdout.write(f"\n\033[1;32m{prompt}\033[0m")
        sys.stdout.flush()
        
        for _ in range(max_new):
            current_indices_list = indices[-self.config['context_length']:]
            pad_len = self.config['context_length'] - len(current_indices_list)
            if pad_len > 0:
                current_indices_list = [self.converter.pad_idx] * pad_len + current_indices_list
            
            context_array = np.array(current_indices_list, dtype=np.int32)
            i_batch = context_array[None, :]
            
            h_batch = self.hasher.hash_sequence(context_array)
            pad_len_h = self.config['context_length'] - len(h_batch)
            if pad_len_h > 0:
                h_batch = np.pad(h_batch, (pad_len_h, 0), 'constant', constant_values=0)
            h_batch = h_batch[None, :]
            
            key, subkey = jax.random.split(key)
            next_idx_array = self.predict_step(self.params, i_batch, h_batch, subkey, temp, top_p)
            
            if not self.jit_compiled:
                next_idx_array.block_until_ready()
                self.jit_compiled = True
            
            new_idx = int(next_idx_array.item())
            if new_idx == self.converter.pad_idx: break
            
            new_char = self.converter.idx_to_char.get(new_idx, '�')
            indices.append(new_idx)
            sys.stdout.write(new_char)
            sys.stdout.flush()
        print()

def interactive_mode(model_basename):
    try: oracle = WubuOracle(model_basename)
    except FileNotFoundError: print(f"\n[ERROR] Model file not found. Train first: python {sys.argv[0]} train"); return
    except Exception: traceback.print_exc(); return
    while True:
        try:
            prompt = input("\nYour Prompt> ")
            if prompt.lower() in ["exit", "quit"]: break
            oracle.generate(prompt)
        except KeyboardInterrupt: print("\n-- Exiting. --"); break
        except Exception as e: print(f"\nAn error occurred: {e}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "infer"]:
        print(f"Usage: python {sys.argv[0]} [train|infer]"); sys.exit(1)
    if sys.argv[1] == "train":
        training_main()
    elif sys.argv[1] == "infer":
        interactive_mode("wubumind_galactic_core_v1")

if __name__ == "__main__":
    main()
