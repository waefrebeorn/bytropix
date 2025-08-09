# %% PYTHON FILE
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import numpy as np
import math
import os
import time
from tqdm import tqdm
import pickle
import json 
from typing import Any, Sequence, Dict, Tuple
from collections import Counter
import unicodedata

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

# --- SOLUTION: The Rite of Distillation ---
def distill_text_from_corpus(data: Any) -> str:
    if isinstance(data, str): return data + "\n"
    elif isinstance(data, dict): return "".join(distill_text_from_corpus(v) for v in data.values())
    elif isinstance(data, list): return "".join(distill_text_from_corpus(item) for item in data)
    return ""

# --- CORRECTED & PURIFIED Vocabulary Creation ---
def create_corpus_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Creates a vocabulary from ALL unique characters in the text, plus PAD and UNK."""
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
        self.unk_idx = self.char_to_idx['<UNK>']
        self.special_vectors = {
            self.char_to_idx['<PAD>']: [0.0] * self.feature_dim,
            self.unk_idx: self.convert_char(ord('?'))
        }
    def get_indices_from_text(self, text:str) -> list[int]:
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]
    def convert_char_from_vocab_idx(self, vocab_idx: int) -> list[float]:
        if vocab_idx in self.special_vectors: return self.special_vectors[vocab_idx]
        char = self.idx_to_char.get(vocab_idx, '?')
        return self.convert_char(ord(char))
    def convert_char(self, char_ord: int) -> list[float]:
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

# --- The rest of the script is unchanged ---
class RollingHasher:
    def __init__(self, window_size, base=313, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus, self.precomputed_base = window_size, base, modulus, pow(base, window_size - 1, modulus)
    def hash_sequence(self, values: list[int]):
        if len(values) < self.window_size: return []
        hashes, current_hash = [], 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + values[i]) % self.modulus
        hashes.append(current_hash)
        for i in range(1, len(values) - self.window_size + 1):
            current_hash = ((current_hash - values[i-1] * self.precomputed_base) * self.base + values[i+self.window_size-1]) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes.append(current_hash)
        return hashes
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
def create_sliding_windows(x, window_size):
    B, H, N, D_h = x.shape; padding = jnp.zeros((B, H, window_size-1, D_h), dtype=x.dtype)
    x_padded = jnp.concatenate([padding, x], axis=2)
    base_indices = jnp.arange(window_size)[None, :]; offsets = jnp.arange(N)[:, None]
    return x_padded[:, :, base_indices + offsets, :]
class WubuBlock(nn.Module):
    dim: int; n_heads: int; attention_window: int; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    def setup(self):
        assert self.dim % self.n_heads == 0; self.h_dim = self.dim // self.n_heads
        self.q_proj, self.k_proj, self.v_proj = (nn.Dense(self.dim,name=f"{n}_proj",dtype=self.dtype,param_dtype=self.param_dtype) for n in "qkv")
        self.out_proj = nn.Dense(self.dim, name="out_proj", dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn = HyperbolicFFN(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.norm1 = nn.LayerNorm(dtype=jnp.float32); self.norm2 = nn.LayerNorm(dtype=jnp.float32)
        self.geo_scale = self.param('geo_scale', nn.initializers.ones, (self.n_heads, 1, 1), self.param_dtype)
    def tangent_space_norm(self, x, norm_layer, c):
        x_tangent = PoincareBall.logmap0(x, c); return PoincareBall.expmap0(norm_layer(x_tangent).astype(self.dtype), c)
    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        x_f32 = x.astype(jnp.float32); x_r, x_i = jnp.split(x_f32, 2, -1); x_c = jax.lax.complex(x_r, x_i)
        freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])[:, -x.shape[1]:, :, :]
        return jnp.concatenate([(x_c * freqs_cis).real, (x_c * freqs_cis).imag], -1).astype(x.dtype)
    def __call__(self, x, freqs_cis, c, cache=None):
        B, N, _ = x.shape; c_bcast_ln = c.reshape(B, 1, 1)
        x_res1 = x; x_norm = self.tangent_space_norm(x, self.norm1, c_bcast_ln)
        x_tangent = PoincareBall.logmap0(x_norm, c_bcast_ln)
        q,k,v = [p(x_tangent).reshape(B,N,self.n_heads,self.h_dim) for p in (self.q_proj,self.k_proj,self.v_proj)]
        q_rot, k_rot = self.apply_rotary_emb(q, freqs_cis), self.apply_rotary_emb(k, freqs_cis)
        c_bcast_attn = c.reshape(B,1,1,1); q_hyper,new_k_hyper = PoincareBall.expmap0(q_rot,c_bcast_attn),PoincareBall.expmap0(k_rot,c_bcast_attn)
        k_full, v_full = (jnp.concatenate([cache[i], t], axis=1) if cache else t for i, t in enumerate([new_k_hyper, v]))
        updated_cache = {'k': k_full, 'v': v_full}
        q_t, k_t, v_t = (t.transpose(0,2,1,3) for t in (q_hyper,k_full,v_full))
        k_w, v_w = create_sliding_windows(k_t, self.attention_window)[:,:,-N:], create_sliding_windows(v_t, self.attention_window)[:,:,-N:]
        attn_dist = PoincareBall.dist(q_t[:,:,:,None,:], k_w, c.reshape(B,1,1,1,1))
        attn_weights = nn.softmax(-self.geo_scale * attn_dist.astype(jnp.float32), axis=-1).astype(self.dtype)
        attn_out_euc = jnp.einsum('bhnw,bhnwd->bhnd', attn_weights, v_w).transpose(0,2,1,3).reshape(B,N,self.dim)
        attn_out_hyp = PoincareBall.expmap0(self.out_proj(attn_out_euc), c_bcast_ln)
        x_attention = PoincareBall.mobius_add(x_res1, attn_out_hyp, c_bcast_ln)
        x_res2 = x_attention; x_norm2 = self.tangent_space_norm(x_attention, self.norm2, c_bcast_ln)
        x_ffn = self.ffn(x_norm2, c_bcast_ln)
        return PoincareBall.mobius_add(x_res2, x_ffn, c_bcast_ln), updated_cache

class WubuMind(nn.Module):
    vocab_size: int; d_model: int; n_heads: int; attention_window: int; modulus: int
    hash_window: int; layers_per_stage: Sequence[int]; downsample_rate: int; rule_embed_dim: int
    max_len: int = 4096; dtype: Any = jnp.bfloat16; param_dtype: Any = jnp.float32
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
        freqs = 1.0/(theta**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))
        return jnp.exp(1j*jnp.outer(jnp.arange(end),freqs))
    @nn.compact
    def __call__(self, hashes, values, cache=None):
        B, N, _ = values.shape; h_dim = self.d_model//self.n_heads
        if not cache: cache = [None]*sum(self.layers_per_stage)
        hash_proj = self.param('hash_projector', nn.initializers.normal(0.02), (1, self.d_model - self.rule_embed_dim), self.param_dtype)
        hash_embed = ((hashes[...,None]/self.modulus).astype(self.dtype)) @ hash_proj
        rule_embed = nn.Dense(self.rule_embed_dim, name="rule_proj")(values.astype(self.dtype))
        combined_euc = jnp.concatenate([hash_embed, rule_embed], axis=-1)
        x_tangent = nn.Dense(self.d_model,dtype=self.dtype,param_dtype=self.param_dtype,name="bridge_proj")(combined_euc)
        c_logits_init=nn.Dense(1,name="c_pred_initial")(jnp.mean(x_tangent,axis=1))
        current_c=nn.softplus(jnp.clip(c_logits_init,-5,5))
        x=PoincareBall.expmap0(x_tangent, current_c.reshape(B,1,1))
        freqs_cis=self.precompute_freqs_cis(h_dim,self.max_len)
        updated_cache, cache_idx = [], 0
        for i, num_layers in enumerate(self.layers_per_stage):
            stage_context=PoincareBall.logmap0(x, current_c.reshape(B,1,1))
            c_logits=nn.Dense(1,name=f"c_pred_{i}")(jnp.mean(stage_context,axis=1))
            c=nn.softplus(jnp.clip(c_logits,-5,5))
            start_pos = cache[cache_idx]['k'].shape[1] if cache[cache_idx] else 0
            stage_freqs_cis=freqs_cis[start_pos:start_pos+x.shape[1]]
            for _ in range(num_layers):
                x, layer_cache=WubuBlock(self.d_model,self.n_heads,self.attention_window,name=f"stage_{i}_block_{_}")(x,stage_freqs_cis,c,cache=cache[cache_idx])
                updated_cache.append(layer_cache); cache_idx+=1
            current_c=c
            if i<len(self.layers_per_stage)-1:
                x_tangent_down=PoincareBall.logmap0(x,current_c.reshape(B,1,1))
                x_tangent_down=LearnedDownsampler(self.d_model,self.downsample_rate,name=f"downsampler_{i}")(x_tangent_down)
                c_logits_down=nn.Dense(1,name=f"c_pred_down_{i}")(jnp.mean(x_tangent_down,axis=1))
                current_c=nn.softplus(jnp.clip(c_logits_down,-5,5))
                x=PoincareBall.expmap0(x_tangent_down,current_c.reshape(B,1,1))
        final_tangent=PoincareBall.logmap0(x,current_c.reshape(B,1,1))
        logits=nn.Dense(self.vocab_size,dtype=jnp.float32,name="output_proj")(final_tangent)
        return logits[:,-1,:], updated_cache

def data_generator(text_corpus, converter, hasher, key, batch_size, context_length, hash_window):
    num_chars = len(text_corpus)
    while True:
        key, perm_key = jax.random.split(key)
        start_indices = jax.random.randint(perm_key, (batch_size,), 0, num_chars - context_length - hash_window)
        h_batch, v_batch, t_batch = [], [], []
        for start_idx in start_indices:
            snippet = text_corpus[start_idx : start_idx + context_length + hash_window]
            indices = converter.get_indices_from_text(snippet)
            hashes = hasher.hash_sequence(indices)
            values = np.array([converter.convert_char_from_vocab_idx(i) for i in indices])
            h_batch.append(hashes[1 : context_length + 1])
            v_batch.append(values[hash_window : context_length + hash_window])
            t_batch.append([indices[context_length + hash_window - 1]])
        yield (jnp.array(h_batch), jnp.array(v_batch), jnp.array(t_batch))

@partial(jax.jit, static_argnames=['static_apply_fn'])
def train_step(state, batch, static_apply_fn):
    hashes, values, targets = batch
    def loss_fn(p):
        logits, _ = static_apply_fn({'params': p}, hashes, values)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets.squeeze()).mean()
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
    state = serialization.from_state_dict(state, save_obj['state'])
    print(f"--- Checkpoint loaded from {filename}, resuming from epoch {save_obj['epoch']} ---")
    return state, save_obj['epoch']
def save_config(config, char_to_idx, basename):
    with open(f"{basename}.json", 'w') as f: json.dump(config, f, indent=4)
    with open(f"{basename}_vocab.json", 'w', encoding='utf-8') as f: json.dump(char_to_idx, f, indent=4)
    print(f"--- Model config and vocab saved to {basename}.json / _vocab.json ---")

def training_main():
    MODEL_BASENAME = "wubumind_v11_absolute_vocab"
    MODEL_CONFIG_BASE = {'hash_window':8, 'd_model':512, 'n_heads':8, 'attention_window':64, 'layers_per_stage':[2,4,2], 'downsample_rate':2, 'modulus':10**9+7}
    TRAINING_CONFIG = {'epochs':25, 'effective_batch_size':128, 'per_device_batch_size':16, 'peak_learning_rate':4e-4, 'warmup_steps':500, 'force_retrain':False, 'context_length': 256}
    key = jax.random.PRNGKey(42)
    print(f"--- WubuMind v11 Foundry (Absolute Vocab Edition) ---\n--- Using device: {jax.devices()[0].platform.upper()} ---")

    try:
        import CORPUS
        full_corpus_data = {
            "SHAKESPEARE_CODICIL": CORPUS.SHAKESPEARE_CODICIL, "WUBU_MANIFESTO": CORPUS.WUBU_MANIFESTO,
            "ALL_CORPUS": CORPUS.ALL_CORPUS, "AI_CORPUS_DIALOGUS_EXEMPLA": CORPUS.AI_CORPUS_DIALOGUS_EXEMPLA,
            "AI_CORPUS_SYSTEMA_INTERNA": CORPUS.AI_CORPUS_SYSTEMA_INTERNA, "AI_CORPUS_ASSISTENTIA": CORPUS.AI_CORPUS_ASSISTENTIA
        }
        corpus_text = distill_text_from_corpus(full_corpus_data)
        print(f"--- Corpus loaded and distilled from IN-MEMORY CORPUS.py object: {len(corpus_text):,} chars. ---")
    except (ImportError, AttributeError):
        print("--- CORPUS.py not found or incomplete. Training on own source code as fallback. ---")
        try:
            with open(__file__, 'r', encoding='utf-8') as f: corpus_text = f.read()
            print(f"--- Corpus loaded from source code: {len(corpus_text):,} chars. ---")
        except Exception: corpus_text = ""
    if not corpus_text: print("FATAL: Corpus text is empty."); return

    char_to_idx, idx_to_char = create_corpus_vocab(corpus_text) # No longer pruned
    converter = UnicodeGeometrodynamicConverter(char_to_idx, idx_to_char)
    MODEL_CONFIG = {**MODEL_CONFIG_BASE, 'vocab_size':len(char_to_idx), 'rule_embed_dim':converter.feature_dim}
    hasher = RollingHasher(MODEL_CONFIG_BASE['hash_window'])
    model = WubuMind(**MODEL_CONFIG)
    key, init_key = jax.random.split(key)
    train_gen_init = data_generator(corpus_text, converter, hasher, key, 1, TRAINING_CONFIG['context_length'], MODEL_CONFIG_BASE['hash_window'])
    init_batch = next(train_gen_init)
    params = model.init(init_key, init_batch[0], init_batch[1])['params']
    print(f'--- Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters. ---')

    num_examples = len(corpus_text) - TRAINING_CONFIG['context_length'] - MODEL_CONFIG_BASE['hash_window']
    steps_per_epoch = max(1, num_examples // TRAINING_CONFIG['effective_batch_size'])
    total_steps = TRAINING_CONFIG['epochs'] * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, TRAINING_CONFIG['peak_learning_rate'], TRAINING_CONFIG['warmup_steps'], max(1, total_steps - TRAINING_CONFIG['warmup_steps']))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=0.01))
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    save_config(MODEL_CONFIG, char_to_idx, MODEL_BASENAME)
    state, start_epoch = load_checkpoint(state, MODEL_BASENAME)
    if start_epoch >= TRAINING_CONFIG['epochs'] and not TRAINING_CONFIG['force_retrain']: print("Training previously completed."); return
    
    train_gen = data_generator(corpus_text, converter, hasher, key, TRAINING_CONFIG['per_device_batch_size'], TRAINING_CONFIG['context_length'], MODEL_CONFIG_BASE['hash_window'])
    grad_accum_steps = TRAINING_CONFIG['effective_batch_size'] // TRAINING_CONFIG['per_device_batch_size']
    epoch = start_epoch
    try:
        start_time = time.time()
        for epoch in range(start_epoch, TRAINING_CONFIG['epochs']):
            with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}") as pbar:
                for _ in pbar:
                    accum_loss = 0.0
                    for _ in range(grad_accum_steps):
                        batch = next(train_gen)
                        state, loss = train_step(state, batch, state.apply_fn)
                        if jnp.isnan(loss): print("\nFATAL: Loss is NaN."); return
                        accum_loss += loss
                    pbar.set_postfix(loss=f"{accum_loss/grad_accum_steps:.4f}")
            save_checkpoint(state, epoch + 1, MODEL_BASENAME)
        print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    except KeyboardInterrupt:
        save_checkpoint(state, epoch, MODEL_BASENAME)

class WubuOracle:
    def __init__(self, model_basename: str):
        print("--- The Oracle is Awakening ---")
        self.basename = model_basename
        with open(f"{self.basename}.json", 'r') as f: self.config = json.load(f)
        with open(f"{self.basename}_vocab.json", 'r', encoding='utf-8') as f: char_to_idx = json.load(f)
        idx_to_char = {int(v): k for k, v in char_to_idx.items()}
        self.converter = UnicodeGeometrodynamicConverter(char_to_idx, idx_to_char)
        self.hasher = RollingHasher(self.config['hash_window'])
        self.model = WubuMind(**self.config)
        self.config['context_length'] = 256 # Add context_length for inference
        with open(f"{self.basename}.pkl", 'rb') as f: save_obj = pickle.load(f)
        dummy_state = train_state.TrainState.create(apply_fn=self.model.apply, params={}, tx=optax.adamw(1e-4))
        self.params = serialization.from_state_dict(dummy_state, save_obj['state']).params
        self.predict_step = self._jit_predict_step()
        print("--- The Oracle is Ready to Speak ---")
    def _prepare_inputs(self, indices: list[int]):
        hashes = jnp.array([self.hasher.hash_sequence(indices)])
        values = np.array([self.converter.convert_char_from_vocab_idx(i) for i in indices])
        return hashes, jnp.array([values])
    def _jit_predict_step(self):
        @partial(jax.jit, static_argnames=['temp', 'top_p'])
        def step_fn(params, hashes, values, cache, key, temp, top_p):
            logits, updated_cache = self.model.apply({'params': params}, hashes, values, cache=cache)
            scaled = logits / jnp.maximum(temp, 1e-6)
            sorted_logits, sorted_indices = jax.lax.sort(scaled, is_stable=False)
            sorted_logits, sorted_indices = sorted_logits[..., ::-1], sorted_indices[..., ::-1]
            cum_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
            sorted_to_remove = cum_probs > top_p
            sorted_to_remove = jnp.concatenate([jnp.zeros_like(sorted_to_remove[..., :1]), sorted_to_remove[..., :-1]], axis=-1)
            to_remove = jnp.zeros_like(sorted_to_remove).at[..., sorted_indices].set(sorted_to_remove)
            scaled = jnp.where(to_remove, -jnp.inf, scaled)
            return jax.random.categorical(key, scaled, axis=-1), updated_cache
        return step_fn
    def generate(self, prompt: str, max_new: int = 500, temp: float = 0.7, top_p: float = 0.95):
        key = jax.random.PRNGKey(int(time.time()))
        indices = self.converter.get_indices_from_text(prompt)
        ctx_len = self.config['context_length']
        if len(indices) < ctx_len: indices = [0] * (ctx_len - len(indices)) + indices
        indices = indices[-ctx_len:]
        print("Oracle is thinking...", end="", flush=True)
        h, v = self._prepare_inputs(indices)
        _, cache = self.model.apply({'params': self.params}, h, v)
        print(" Done.")
        for _ in tqdm(range(max_new), desc="Generating"):
            key, subkey = jax.random.split(key)
            h, v = self._prepare_inputs(indices[-ctx_len:])
            next_idx, cache = self.predict_step(self.params, h, v, cache, subkey, temp, top_p)
            new_idx = int(next_idx.item())
            indices.append(new_idx)
            yield self.converter.idx_to_char.get(new_idx, '�')

def interactive_mode(model_basename):
    try: oracle = WubuOracle(model_basename)
    except FileNotFoundError: print(f"\n[ERROR] Model file not found. Train first: python {sys.argv[0]} train"); return
    except Exception as e: print(f"\nAn unexpected error occurred during init: {e}"); return
    while True:
        try:
            prompt = input("\nYour Prompt> ")
            if prompt.lower() in ["exit", "quit"]: break
            print("WubuOracle> ", end="")
            for char in oracle.generate(prompt): print(char, end="", flush=True)
        except KeyboardInterrupt: print("\n-- Generation Interrupted --"); continue
        except Exception as e: print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "infer"]:
        print(f"Usage: python {sys.argv[0]} [train|infer]")
        sys.exit(1)
    if sys.argv[1] == "train": training_main()
    elif sys.argv[1] == "infer": interactive_mode("wubumind_v11_absolute_vocab")
