# %% PYTHON FILE
import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import remat # Import remat for gradient checkpointing
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import numpy as np
import math
import time
from tqdm import tqdm
import pickle
import json
from typing import Any, Sequence, Dict, Tuple
from collections import Counter
import unicodedata
import sys
import dataclasses

# Ensure full GPU memory is available for JAX.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_debug_nans", False)

# --- Unicode Pre-computation Rite (Unchanged) ---
print("--- Performing one-time Unicode Pre-computation Rite (StdLib Only)... ---")
SCRIPT_GROUPS = set()
for i in range(0x110000):
    try:
        name = unicodedata.name(chr(i))
        first_word = name.split(' ')[0]
        SCRIPT_GROUPS.add(first_word)
    except (ValueError, TypeError): continue
ALL_SCRIPTS = sorted(list(SCRIPT_GROUPS))
SCRIPT_TO_IDX = {name: i for i, name in enumerate(ALL_SCRIPTS)}
NUM_SCRIPTS = len(ALL_SCRIPTS)
ALL_CATEGORIES = sorted(list(set(unicodedata.category(chr(i)) for i in range(0x110000))))
CATEGORY_TO_IDX = {name: i for i, name in enumerate(ALL_CATEGORIES)}
NUM_CATEGORIES = len(ALL_CATEGORIES)
PRINTABLE_CATEGORIES = {'L', 'M', 'N', 'P', 'S'}
print("--- Rite complete. Knowledge assimilated from standard library. ---")

def distill_text_from_corpus(data: Any) -> str:
    if isinstance(data, str): return data + "\n"
    elif isinstance(data, dict): return "".join(distill_text_from_corpus(v) for v in data.values())
    elif isinstance(data, list): return "".join(distill_text_from_corpus(item) for item in data)
    return ""

def create_corpus_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    print(f"--- Discovering absolute vocabulary from corpus... ---")
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
        self.feature_dim = 9
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        self.unk_idx = self.char_to_idx.get('<UNK>', 1)
        self.pad_idx = self.char_to_idx.get('<PAD>', 0)
    def get_indices_from_text(self, text:str) -> list[int]:
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]
    def _convert_char(self, char_ord: int) -> list[float]:
        try: char = chr(char_ord)
        except ValueError: return [0.0] * self.feature_dim
        try:
            name = unicodedata.name(char)
            script_group = name.split(' ')[0]
            script_idx = SCRIPT_TO_IDX.get(script_group, -1)
        except ValueError: script_idx = -1
        normalized_script = script_idx / (NUM_SCRIPTS - 1) if NUM_SCRIPTS > 1 and script_idx != -1 else 0.0
        category = unicodedata.category(char)
        category_idx = CATEGORY_TO_IDX.get(category, -1)
        normalized_category = category_idx / (NUM_CATEGORIES-1) if NUM_CATEGORIES > 1 and category_idx != -1 else 0.0
        utf8_bytes = char.encode('utf-8', errors='ignore')
        byte_count = len(utf8_bytes) / 4.0
        leading_byte = utf8_bytes[0] / 255.0 if utf8_bytes else 0.0
        numeric_val = unicodedata.numeric(char, 0.0)
        case_val = 1.0 if category.startswith('Lu') else (-1.0 if category.startswith('Ll') else 0.0)
        is_printable = 1.0 if category[0] in PRINTABLE_CATEGORIES else 0.0
        return [
            normalized_script, normalized_category, byte_count, leading_byte,
            1.0 if category == 'Nd' else 0.0, math.log(1 + numeric_val) / 10.0,
            case_val, is_printable, char_ord / 0x10FFFF
        ]
    def get_feature_vector(self, vocab_idx: int) -> list[float]:
        if vocab_idx == self.pad_idx: return [0.0] * self.feature_dim
        if vocab_idx == self.unk_idx: return self._convert_char(ord('?'))
        char = self.idx_to_char.get(vocab_idx, '?')
        return self._convert_char(ord(char))
    def precompute_lookup_table(self) -> np.ndarray:
        print("--- Pre-computing vectorized feature lookup table... ---")
        table = np.zeros((self.vocab_size, self.feature_dim), dtype=np.float32)
        for i in range(self.vocab_size):
            table[i] = self.get_feature_vector(i)
        print("--- Lookup table created. ---")
        return table

class RollingHasher:
    def __init__(self, window_size, base=313, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus, self.precomputed_base = window_size, base, modulus, pow(base, window_size - 1, modulus)
    def hash_sequence(self, values: np.ndarray):
        num_values = len(values)
        if num_values < self.window_size: return np.array([], dtype=np.int32)
        hashes = []
        current_hash = 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + int(values[i])) % self.modulus
        hashes.append(current_hash)
        for i in range(1, num_values - self.window_size + 1):
            current_hash = ((current_hash - int(values[i-1]) * self.precomputed_base) * self.base + int(values[i+self.window_size-1])) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes.append(current_hash)
        return np.array(hashes, dtype=np.int32)

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
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS); add_xy = PoincareBall.mobius_add(-x, y, c)
        add_norm = jnp.linalg.norm(add_xy, axis=-1)
        arg = jnp.minimum(sqrt_c.squeeze(-1) * add_norm, 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg) / sqrt_c.squeeze(-1)

class HyperbolicFFN(nn.Module):
    dim: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, c):
        x_tangent = PoincareBall.logmap0(x, c)
        ffn_output_tangent = nn.Sequential([nn.Dense(self.dim*4,dtype=self.dtype,param_dtype=self.param_dtype), nn.gelu, nn.Dense(self.dim,dtype=self.dtype,param_dtype=self.param_dtype)])(x_tangent)
        return PoincareBall.expmap0(ffn_output_tangent, c)

class LearnedDownsampler(nn.Module):
    dim: int; rate: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, (self.rate,), (self.rate,), 'VALID', name="downsampler_conv", dtype=self.dtype, param_dtype=self.param_dtype)(x)

class WubuBlock(nn.Module):
    dim: int; n_heads: int; block_size: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    
    def setup(self):
        assert self.dim % self.n_heads == 0; self.h_dim = self.dim // self.n_heads
        self.qkv_proj = nn.Dense(self.dim * 3, name="qkv_proj", dtype=self.dtype, param_dtype=self.param_dtype)
        self.out_proj = nn.Dense(self.dim, name="out_proj", dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn = HyperbolicFFN(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.norm1 = nn.LayerNorm(dtype=jnp.float32); self.norm2 = nn.LayerNorm(dtype=jnp.float32)
        self.c_per_head_logits = self.param('c_per_head_logits', nn.initializers.zeros, (self.n_heads,), self.param_dtype)
        self.geo_scale = self.param('geo_scale', nn.initializers.ones, (1, self.n_heads, 1, 1), self.param_dtype)

    def tangent_space_norm(self, x, norm_layer, c):
        x_tangent = PoincareBall.logmap0(x, c); return PoincareBall.expmap0(norm_layer(x_tangent).astype(self.dtype), c)

    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        x_f32 = x.astype(jnp.float32)
        x_r, x_i = jnp.split(x_f32, 2, -1)
        x_c = jax.lax.complex(x_r, x_i)
        freqs_cis = freqs_cis.reshape(1, 1, freqs_cis.shape[0], freqs_cis.shape[1])
        x_rotated = x_c * freqs_cis
        return jnp.concatenate([x_rotated.real, x_rotated.imag], -1).astype(x.dtype)

    @nn.compact
    def __call__(self, x, freqs_cis, c_global, cache, start_pos):
        B, N, _ = x.shape
        assert N % self.block_size == 0, f"Sequence length {N} must be divisible by block_size {self.block_size}"
        num_blocks = N // self.block_size
        B_eff = B * num_blocks

        x_blocks = x.reshape(B_eff, self.block_size, self.dim)
        
        x_res1 = x_blocks
        c_bcast = jnp.repeat(c_global, num_blocks, axis=0).reshape(B_eff, 1, 1)
        x_norm = self.tangent_space_norm(x_blocks, self.norm1, c_bcast)
        x_tangent = PoincareBall.logmap0(x_norm, c_bcast)

        qkv = self.qkv_proj(x_tangent)
        qkv = qkv.reshape(B_eff, self.block_size, 3, self.n_heads, self.h_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        freqs_cis_h = WubuMind.precompute_freqs_cis(self.h_dim, self.block_size)
        q_rot = self.apply_rotary_emb(q, freqs_cis_h)
        k_rot = self.apply_rotary_emb(k, freqs_cis_h)
        
        c_per_head = nn.softplus(self.c_per_head_logits).reshape(1, self.n_heads, 1, 1)
        
        q_hyp = PoincareBall.expmap0(q_rot, c_per_head)
        k_hyp = PoincareBall.expmap0(k_rot, c_per_head)

        # Reshape c_per_head to be 5D for the dist calculation, matching the 5D intermediate tensors
        c_for_dist = c_per_head[:, :, None, :, :]
        
        dist = PoincareBall.dist(q_hyp[:, :, :, None, :], k_hyp[:, :, None, :, :], c_for_dist)
        attn_weights = nn.softmax(-self.geo_scale * dist.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v).transpose(0,2,1,3).reshape(B_eff, self.block_size, self.dim)
        
        attn_out_proj = self.out_proj(attn_out)
        attn_out_hyp = PoincareBall.expmap0(attn_out_proj, c_bcast)
        x_attention = PoincareBall.mobius_add(x_res1, attn_out_hyp, c_bcast)
        
        x_res2 = x_attention
        x_norm2 = self.tangent_space_norm(x_attention, self.norm2, c_bcast)
        x_ffn = self.ffn(x_norm2, c_bcast)
        
        x_final_blocks = PoincareBall.mobius_add(x_res2, x_ffn, c_bcast)
        
        return x_final_blocks.reshape(B, N, self.dim), None

@dataclasses.dataclass
class WubuMind(nn.Module):
    vocab_size: int; d_model: int; n_heads: int; block_size: int; modulus: int
    hash_window: int; layers_per_stage: Sequence[int]; downsample_rate: int; rule_embed_dim: int
    max_len: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
        freqs = 1.0/(theta**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))
        return jnp.exp(1j*jnp.outer(jnp.arange(end),freqs))

    @nn.compact
    def __call__(self, hashes, values, cache, start_pos):
        B, N, _ = values.shape
        hash_proj = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model - self.rule_embed_dim), self.param_dtype)
        hash_embed = (hashes[...,None].astype(self.dtype) / self.modulus) @ hash_proj
        rule_embed = nn.Dense(self.rule_embed_dim, name="rule_proj")(values.astype(self.dtype))
        combined_euc = jnp.concatenate([hash_embed, rule_embed], axis=-1)
        x_tangent = nn.Dense(self.d_model,dtype=self.dtype,param_dtype=self.param_dtype,name="bridge_proj")(combined_euc)

        c_logits_init=nn.Dense(1,name="c_pred_initial")(jnp.mean(x_tangent,axis=1))
        current_c=nn.softplus(jnp.clip(c_logits_init,-5,5))
        x=PoincareBall.expmap0(x_tangent, current_c.reshape(B,1,1))
        
        freqs_cis_full = self.precompute_freqs_cis(self.d_model, self.max_len)
        
        current_pos_in_stage = start_pos
        for i, num_layers in enumerate(self.layers_per_stage):
            stage_context=PoincareBall.logmap0(x, current_c.reshape(B,1,1))
            c_logits=nn.Dense(1,name=f"c_pred_{i}")(jnp.mean(stage_context,axis=1))
            c_global_stage=nn.softplus(jnp.clip(c_logits,-5,5))
            
            stage_freqs_cis = jax.lax.dynamic_slice(freqs_cis_full, (current_pos_in_stage, 0), (x.shape[1], freqs_cis_full.shape[1]))
            
            for l_idx in range(num_layers):
                rematted_block_class = remat(WubuBlock, static_argnums=(3,4))
                block_instance = rematted_block_class(self.d_model, self.n_heads, self.block_size, name=f"stage_{i}_block_{l_idx}")
                x, _ = block_instance(x, stage_freqs_cis, c_global_stage, None, current_pos_in_stage)

            current_c = c_global_stage
            
            if i < len(self.layers_per_stage)-1:
                if x.shape[1] > 1:
                    x_tangent_down=PoincareBall.logmap0(x,current_c.reshape(B,1,1))
                    x_tangent_down=LearnedDownsampler(self.d_model,self.downsample_rate,name=f"downsampler_{i}")(x_tangent_down)
                    c_logits_down=nn.Dense(1,name=f"c_pred_down_{i}")(jnp.mean(x_tangent_down,axis=1))
                    current_c=nn.softplus(jnp.clip(c_logits_down,-5,5))
                    x=PoincareBall.expmap0(x_tangent_down,current_c.reshape(B,1,1))
                    current_pos_in_stage = 0
                    
        final_tangent=PoincareBall.logmap0(x,current_c.reshape(B,1,1))
        logits=nn.Dense(self.vocab_size,dtype=jnp.float32,name="output_proj")(final_tangent)
        
        return logits, None
        
def prepare_training_data_on_device(text_corpus, converter, hasher, context_length, hash_window, batch_size):
    print("--- Beginning high-performance data pipeline... ---")
    lookup_table = converter.precompute_lookup_table()
    print("--- Converting corpus to indices... ---")
    indices = np.array(converter.get_indices_from_text(text_corpus), dtype=np.int32)
    print("--- Pre-computing all hashes... ---")
    hashes = hasher.hash_sequence(indices)
    print("--- Assembling sequences with sliding windows... ---")
    num_samples = len(hashes) - context_length + 1
    strides = hashes.strides
    all_hashes = np.lib.stride_tricks.as_strided(hashes, shape=(num_samples, context_length), strides=(strides[0], strides[0]))
    strides = indices.strides
    all_targets = np.lib.stride_tricks.as_strided(indices[hash_window:], shape=(num_samples, context_length), strides=(strides[0], strides[0]))
    all_values_indices = np.lib.stride_tricks.as_strided(indices[hash_window-1:-1], shape=(num_samples, context_length), strides=(strides[0], strides[0]))
    num_batches = num_samples // batch_size
    num_to_trim = num_samples % batch_size
    if num_to_trim > 0: all_hashes, all_targets, all_values_indices = [arr[:-num_to_trim] for arr in (all_hashes, all_targets, all_values_indices)]
    all_hashes_batched = all_hashes.reshape(num_batches, batch_size, context_length)
    all_targets_batched = all_targets.reshape(num_batches, batch_size, context_length)
    print("--- Performing vectorized value lookup... ---")
    all_values_batched = lookup_table[all_values_indices].reshape(num_batches, batch_size, context_length, -1)
    print(f"--- Data preparation complete. {num_batches} batches created. ---")
    print("--- Transferring all batches to device... ---")
    device_batches = jax.device_put((all_hashes_batched, all_values_batched.astype(jnp.bfloat16), all_targets_batched))
    print("--- Data pipeline complete. All data is now on-device. ---")
    return device_batches, num_batches

@partial(jax.jit, static_argnames=['static_apply_fn'])
def train_step(state, batch, static_apply_fn):
    hashes, values, targets = batch
    def loss_fn(p):
        logits, _ = static_apply_fn({'params': p}, hashes, values, None, 0)
        target_len = logits.shape[1]
        aligned_targets = targets[:, -target_len:]
        return optax.softmax_cross_entropy_with_integer_labels(logits, aligned_targets).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def save_checkpoint(state, epoch, basename):
    with open(f"{basename}.pkl", 'wb') as f: pickle.dump({'state': serialization.to_state_dict(state), 'epoch': epoch}, f)
    print(f"\n--- Checkpoint saved at epoch {epoch} to {basename}.pkl ---")

def load_checkpoint(state, basename):
    filename = f"{basename}.pkl"
    if not os.path.exists(filename): return state, 0
    with open(filename, 'rb') as f: save_obj = pickle.load(f)
    try:
        print(f"--- Checkpoint found. Restoring parameters from epoch {save_obj['epoch']}... ---")
        if 'params' not in save_obj['state']: save_obj['state'] = {'params': save_obj['state']}
        restored_params = serialization.from_state_dict(state.params, save_obj['state']['params'])
        if 'opt_state' not in save_obj['state']:
            print("--- Optimizer state not found in checkpoint. Resetting optimizer. ---")
            return state.replace(params=restored_params), save_obj['epoch']
        restored_state = serialization.from_state_dict(state, save_obj['state'])
        print("--- Full training state restored successfully. ---")
        return restored_state, save_obj['epoch']
    except Exception as e:
        print(f"--- FAILED to load checkpoint: {e}. Training from scratch. ---")
        if os.path.exists(filename): os.remove(filename)
        return state, 0

def save_config(config, char_to_idx, basename):
    with open(f"{basename}.json", 'w') as f: json.dump(config, f, indent=4)
    with open(f"{basename}_vocab.json", 'w', encoding='utf-8') as f: json.dump(char_to_idx, f, indent=4)
    print(f"--- Model config and vocab saved to {basename}.json / _vocab.json ---")

def training_main():
    MODEL_BASENAME = "wubumind_v11_absolute_vocab"
    MODEL_CONFIG_BASE = {'hash_window':8, 'd_model':256, 'n_heads':4, 'block_size':64, 'layers_per_stage':[4,4,4], 'downsample_rate':2, 'modulus':10**9+7, 'max_len': 4096}
    TRAINING_CONFIG = {'epochs':20, 'batch_size':16, 'peak_learning_rate':5e-4, 'warmup_steps':500, 'force_retrain':False, 'context_length': 256}
    key = jax.random.PRNGKey(42)
    print(f"--- WubuMind v11 Foundry (Absolute Vocab Edition) ---\n--- Using device: {jax.devices()[0].platform.upper()} ---")

    try:
        import CORPUS
        all_found_corpuses = []
        print("--- Dynamically discovering corpuses from CORPUS.py ---")
        for name in dir(CORPUS):
            if not name.startswith('__'):
                obj = getattr(CORPUS, name)
                if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
                    print(f"--- Found and assimilated corpus: {name} ---")
                    all_found_corpuses.extend(obj)
        corpus_text = distill_text_from_corpus(all_found_corpuses)
        print(f"--- Total corpus size: {len(corpus_text):,} chars. ---")
    except ImportError:
        print("--- CORPUS.py not found. Training on own source code as fallback. ---")
        with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read()

    if not corpus_text: print("FATAL: Corpus text is empty."); return

    char_to_idx, idx_to_char = create_corpus_vocab(corpus_text)
    converter = UnicodeGeometrodynamicConverter(char_to_idx, idx_to_char)
    
    ARCH_CONFIG = {**MODEL_CONFIG_BASE, 'vocab_size':len(char_to_idx), 'rule_embed_dim':converter.feature_dim}
    if 'attention_window' in ARCH_CONFIG: del ARCH_CONFIG['attention_window']

    FULL_CONFIG = {**ARCH_CONFIG, 'context_length':TRAINING_CONFIG['context_length']}
    hasher = RollingHasher(ARCH_CONFIG['hash_window'])
    model = WubuMind(**ARCH_CONFIG)
    
    (h_all, v_all, t_all), steps_per_epoch = prepare_training_data_on_device(corpus_text, converter, hasher, TRAINING_CONFIG['context_length'], ARCH_CONFIG['hash_window'], TRAINING_CONFIG['batch_size'])
    
    key, init_key = jax.random.split(key)
    init_batch = (h_all[0], v_all[0])
    params = model.init(init_key, *init_batch, None, 0)['params']
    print(f'--- Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters. ---')

    total_steps = TRAINING_CONFIG['epochs'] * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, TRAINING_CONFIG['peak_learning_rate'], TRAINING_CONFIG['warmup_steps'], total_steps)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    save_config(FULL_CONFIG, char_to_idx, MODEL_BASENAME)
    state, start_epoch = load_checkpoint(state, MODEL_BASENAME)
    if start_epoch >= TRAINING_CONFIG['epochs'] and not TRAINING_CONFIG['force_retrain']: print("Training previously completed."); return
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, TRAINING_CONFIG['epochs']):
            key, shuffle_key = jax.random.split(key)
            shuffled_indices = jax.random.permutation(shuffle_key, steps_per_epoch)
            with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}") as pbar:
                for step_idx in pbar:
                    batch_idx = shuffled_indices[step_idx]
                    batch = (h_all[batch_idx], v_all[batch_idx], t_all[batch_idx])
                    state, loss = train_step(state, batch, state.apply_fn)
                    if jnp.isnan(loss): print("\nFATAL: Loss is NaN."); return
                    pbar.set_postfix(loss=f"{loss:.4f}")
            save_checkpoint(state, epoch+1, MODEL_BASENAME)
    except KeyboardInterrupt:
        save_checkpoint(state, epoch, MODEL_BASENAME)
        print("\n--- Training interrupted by user. Checkpoint saved. ---")

@partial(jax.jit, static_argnames=['model_apply_fn', 'temp', 'top_p'])
def predict_step_fn(model_apply_fn, params, hashes, values, cache, start_pos, key, temp, top_p):
    logits, _ = model_apply_fn({'params': params}, hashes, values, cache, start_pos)
    scaled = logits[:,-1,:] / jnp.maximum(temp, 1e-6)
    sorted_indices = jnp.argsort(scaled, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(scaled, sorted_indices, axis=-1)
    cum_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_to_remove = cum_probs > top_p
    sorted_to_remove = jnp.concatenate([jnp.zeros_like(sorted_to_remove[..., :1]), sorted_to_remove[..., :-1]], axis=-1)
    to_remove = jnp.zeros_like(sorted_to_remove).at[..., sorted_indices].set(sorted_to_remove)
    scaled = jnp.where(to_remove, -jnp.inf, scaled)
    return jax.random.categorical(key, scaled, axis=-1), None

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
        self.feature_table = jax.device_put(self.converter.precompute_lookup_table())
        
        print("--- Assimilating knowledge from checkpoint... ---")
        with open(f"{self.basename}.pkl", 'rb') as f: save_obj = pickle.load(f)
        
        key = jax.random.PRNGKey(0)
        init_len = self.config['context_length']
        dummy_h = jnp.zeros((1, init_len), dtype=np.int32)
        dummy_v = jnp.zeros((1, init_len, self.config['rule_embed_dim']), dtype=np.float32)
        target_params = self.model.init(key, dummy_h, dummy_v, None, 0)['params']
        self.params = serialization.from_state_dict(target=target_params, state=save_obj['state']['params'])
        print(f"--- Oracle has assimilated knowledge from epoch {save_obj['epoch']}. Ready to Speak. ---")

    def generate(self, prompt: str, max_new: int = 500, temp: float = 0.7, top_p: float = 0.95):
        print("\n[INFO] Oracle generation starting...")
        key = jax.random.PRNGKey(int(time.time()))
        indices = self.converter.get_indices_from_text(prompt)
        
        for i in tqdm(range(max_new), desc="Generating"):
            context_len = self.config['context_length']
            
            # Prepare context for this step, padding if necessary
            current_sequence = indices[-context_len:]
            pad_len_seq = context_len - len(current_sequence)
            if pad_len_seq > 0:
                current_sequence = [self.converter.pad_idx] * pad_len_seq + current_sequence
            
            sub_indices = np.array(current_sequence, dtype=np.int32)
            
            hashes = self.hasher.hash_sequence(sub_indices)
            values = self.feature_table[sub_indices]
            
            h_step_b = jnp.array([hashes])
            v_step_b = jnp.array([values])
            
            key, subkey = jax.random.split(key)
            
            # Use the predict_step_fn for sampling
            next_idx_array, _ = predict_step_fn(self.model.apply, {'params': self.params}, h_step_b, v_step_b, None, 0, subkey, temp, top_p)
            
            new_idx = int(next_idx_array.item())
            if new_idx == self.converter.pad_idx and not indices:
                continue
            if new_idx == self.converter.pad_idx:
                print("[INFO] Pad token generated, stopping.")
                break
                
            indices.append(new_idx)
            yield self.converter.idx_to_char.get(new_idx, '�')

def interactive_mode(model_basename):
    try:
        oracle = WubuOracle(model_basename)
    except FileNotFoundError:
        print(f"\n[ERROR] Model file not found. Train first: python {sys.argv[0]} train")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred during init: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            prompt = input("\nYour Prompt> ")
            if prompt.lower() in ["exit", "quit"]: break
            print("WubuOracle> ", end="")
            for char in oracle.generate(prompt): 
                print(char, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n-- Generation Interrupted. Exiting. --")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "infer"]:
        print(f"Usage: python {sys.argv[0]} [train|infer]")
        sys.exit(1)

    if sys.argv[1] == "train": training_main()
    elif sys.argv[1] == "infer": interactive_mode("wubumind_v11_absolute_vocab")
