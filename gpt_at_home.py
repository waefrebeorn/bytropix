import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
except NameError:
    cache_dir = os.path.join(os.path.expanduser("~"), ".jax_cache_ascended_v3")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
import sys
import time
from pathlib import Path
import math
import mmap
import signal
import threading
import platform
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any, Deque, List
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax import struct, serialization
import chex
print("--- Verifying Dependencies ---")
dependencies = [("tensorflow", "tensorflow"), ("tensorflow_datasets", "tensorflow-datasets"),
                ("rich.console", "rich"), ("nvidia.ml", "nvidia-ml-py")]
missing = [pkg for mod, pkg in dependencies if not __import__(mod.split('.')[0], fromlist=['_'] * (mod.count('.') + 1))]
if missing:
    print(f"\n[FATAL] Missing dependencies. Please run: pip install {' '.join(missing)}")
    sys.exit(1)
print("--- All dependencies verified. ---")
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from rich.live import Live; from rich.table import Table; from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
from rich.text import Text; from rich.padding import Padding
import pynvml; pynvml.nvmlInit()
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)

class Config:
    DATA_DIR = "./data"; RAW_TEXT_FILE = "open_orca_formatted.txt"; BASENAME = "AscendedThinker_v3_Jax"; CHECKPOINT_DIR = "./checkpoints"
    ### CRITICAL FIX: The vocabulary size MUST be 256 to match the raw byte data (0-255). ###
    VOCAB_SIZE = 256 
    EFFECTIVE_VOCAB_SIZE = 256

    EPOCHS = 30000; LEARNING_RATE = 1e-4; USE_BFLOAT16 = True; SAVE_EVERY = 2000; FRESH_START = False
    
    ### PERFORMANCE TWEAK: Increased model size to better utilize the GPU. ###
    D_MODEL = 1024; NUM_LAYERS = 4; NUM_HEADS = 8; D_FF = D_MODEL * 4; DROPOUT = 0.1
    
    USE_COGNITIVE_REGULARIZATION = True; COGNITIVE_LOSS_WEIGHT = 0.01; COGNITIVE_LOSS_WARMUP_STEPS = 2000
    USE_GRAMMAR_FIELD = True; NUM_GRAMMAR_EXPERTS = 16
    USE_GEODESIC_REG = True; GEODESIC_LOSS_WEIGHT = 0.01
    USE_COHERENCE_REG = True; COHERENCE_LOSS_WEIGHT = 0.01
    NUM_GROUPS = 4; EMA_DECAY = 0.999
    USE_CANVAS_TRAINING = True
    CANVAS_WIDTH = 64
    CANVAS_HEIGHT = 64
    NUM_CANVAS_PAGES = 5
    BATCH_SIZE = 1; SUPER_BATCH_SIZE = 1
    # Special tokens are repurposed from the 0-255 byte range.
    VOID_TOKEN_ID = 251; MASK_TOKEN_ID = 252; EXPAND_TOKEN_ID = 253; CONTINUE_FWD_TOKEN_ID = 250
    CONTINUE_BWD_TOKEN_ID = 249; SEMANTIC_CHECKSUM_TOKEN_ID = 248
    TEMPERATURE = 1.0; TOP_K = 20
    PREVIEW_EVERY_N_STEPS = 1000
    VOID_CHAR = '░'; MASK_CHAR = '█'; EXPAND_CHAR = '⊕'; CONTINUE_FWD_CHAR = '▷'; CONTINUE_BWD_CHAR = '◁'; SEMANTIC_CHECKSUM_CHAR = '❖'

def toroidal_gradient_transform() -> optax.GradientTransformation:
    def init_fn(params): return optax.EmptyState()
    def update_fn(updates, state, params=None):
        boundary = 2 * jnp.pi
        def wrap_gradient(g): return jnp.mod(g + jnp.pi, boundary) - jnp.pi
        wrapped_updates = jax.tree_util.tree_map(wrap_gradient, updates)
        return wrapped_updates, state
    return optax.GradientTransformation(init_fn, update_fn)

class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock(); self.force_save = False; self.shutdown_event = threading.Event()
        self.is_prompting = False; self.current_prompt_text = ""; self.prompt_submitted = False
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def get_submitted_prompt(self):
        with self.lock:
            if self.prompt_submitted:
                self.prompt_submitted = False; prompt = self.current_prompt_text; self.current_prompt_text = ""; return prompt
            return None
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [p] New Prompt | [s] Force Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt
    else: import select, sys, tty, termios; fd, old_settings = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        if platform.system() != "Windows": tty.setcbreak(sys.stdin.fileno())
        while not shared_state.shutdown_event.is_set():
            if (platform.system() == "Windows" and msvcrt.kbhit()) or (platform.system() != "Windows" and select.select([sys.stdin],[],[],0.05)[0]):
                key_bytes = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(1)
                try: key = key_bytes.decode('utf-8')
                except (UnicodeDecodeError, AttributeError): key = repr(key_bytes)
                with shared_state.lock:
                    if shared_state.is_prompting:
                        if key in ['\r', '\n', '\r\n']: shared_state.is_prompting = False; shared_state.prompt_submitted = True
                        elif key in ["b'\\x7f'", "'\\x7f'", "b'\\b'", "'\\b'"]: shared_state.current_prompt_text = shared_state.current_prompt_text[:-1]
                        elif key.isprintable(): shared_state.current_prompt_text += key
                    else:
                        if key in ['q', '\x03']: shared_state.set_shutdown(); break
                        elif key == 's': shared_state.force_save = True
                        elif key == 'p': shared_state.is_prompting = True
            else: time.sleep(0.05)
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

@struct.dataclass
class CognitiveMetrics: chaos_metric: chex.Array; kl_divergence: chex.Array
@struct.dataclass
class LossComponents: loss_chaos: chex.Array; loss_divergence: chex.Array
class CustomTrainState(train_state.TrainState):
    ema_params: Any
    dropout_key: chex.PRNGKey
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

class GrammarFieldOperator(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        cfg = self.config
        gate_logits = nn.Dense(features=cfg.NUM_GRAMMAR_EXPERTS, use_bias=False, dtype=self.dtype, name="gate")(x)
        weights = nn.softmax(gate_logits, axis=-1)
        transformed_x = nn.Dense(features=cfg.NUM_GRAMMAR_EXPERTS * cfg.D_MODEL, dtype=self.dtype, name="experts")(x)
        transformed_x = transformed_x.reshape(*x.shape[:2], cfg.NUM_GRAMMAR_EXPERTS, cfg.D_MODEL)
        output = jnp.einsum('blk,blkd->bld', weights, transformed_x)
        return x + output

class SingleGroupProcessor(nn.Module):
    config: Config; d_group: int; num_heads_group: int; dtype: Any
    @nn.compact
    def __call__(self, group_x, deterministic: bool):
        cfg = self.config
        norm_x = nn.LayerNorm(dtype=self.dtype)(group_x)
        attn_output = nn.SelfAttention(num_heads=self.num_heads_group, qkv_features=self.d_group, dropout_rate=cfg.DROPOUT, deterministic=deterministic, dtype=self.dtype)(norm_x)
        group_x += attn_output
        norm_x = nn.LayerNorm(dtype=self.dtype)(group_x)
        ffn_output = nn.Sequential([
            nn.Dense(self.d_group * 4, dtype=self.dtype), nn.gelu,
            nn.Dense(self.d_group, dtype=self.dtype), nn.Dropout(cfg.DROPOUT, deterministic=deterministic)
        ])(norm_x)
        return group_x + ffn_output

class GroupedTransformerBlock(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, x, deterministic=False):
        cfg = self.config; assert cfg.D_MODEL % cfg.NUM_GROUPS == 0 and cfg.NUM_HEADS % cfg.NUM_GROUPS == 0
        d_group, num_heads_group = cfg.D_MODEL // cfg.NUM_GROUPS, cfg.NUM_HEADS // cfg.NUM_GROUPS
        B, L, _ = x.shape
        x_grouped = x.reshape(B, L, cfg.NUM_GROUPS, d_group)
        VmappedProcessor = nn.vmap(SingleGroupProcessor, in_axes=(2, None), out_axes=2, variable_axes={'params': 0}, split_rngs={'params': True, 'dropout': True})
        processed_groups = VmappedProcessor(config=cfg, d_group=d_group, num_heads_group=num_heads_group, dtype=self.dtype, name="GroupProcessor")(x_grouped, deterministic)
        x = processed_groups.reshape(B, L, cfg.D_MODEL)
        if cfg.USE_GRAMMAR_FIELD:
            x = GrammarFieldOperator(config=cfg, dtype=self.dtype, name="GrammarField")(x)
        return x

class EntropixEngine(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, hidden_states: chex.Array, logits: chex.Array) -> Tuple[chex.Array, CognitiveMetrics, LossComponents]:
        probs = nn.softmax(logits.astype(jnp.float32), axis=-1); log_probs = nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1, keepdims=True)
        variance = jnp.sum(probs * jnp.power(log_probs + entropy, 2), axis=-1, keepdims=True)
        delta = nn.tanh(variance) * jnp.pi; chi = nn.tanh(entropy) * (jnp.pi / 4.0)
        t_co_real = jnp.cos(delta / 2.0); t_co_imag = jnp.sin(delta / 2.0) * jnp.sin(2.0 * chi)
        h_dim = hidden_states.shape[-1]; h1, h2 = hidden_states[..., :h_dim//2], hidden_states[..., h_dim//2:]
        modulated_h1 = h1 * t_co_real - h2 * t_co_imag; modulated_h2 = h1 * t_co_imag + h2 * t_co_real
        modulated_hidden_states = jnp.concatenate([modulated_h1, modulated_h2], axis=-1).astype(self.dtype)
        chaos_metric = entropy.squeeze(-1) * variance.squeeze(-1)
        kl_div = jnp.sum(probs * (log_probs - jnp.log(1.0/self.config.EFFECTIVE_VOCAB_SIZE)), axis=-1)
        metrics = CognitiveMetrics(chaos_metric=chaos_metric, kl_divergence=kl_div)
        loss_components = LossComponents(loss_chaos=jnp.mean(metrics.chaos_metric), loss_divergence=jnp.mean(metrics.kl_divergence))
        return modulated_hidden_states, metrics, loss_components

class MaskedDenoisingModel(nn.Module):
    config: Config; dtype: Any
    def setup(self):
        cfg = self.config
        self.token_embedding = nn.Embed(cfg.EFFECTIVE_VOCAB_SIZE, cfg.D_MODEL, dtype=self.dtype)
        self.transformer_blocks = [GroupedTransformerBlock(config=cfg, dtype=self.dtype, name=f"block_{i}") for i in range(cfg.NUM_LAYERS)]
        if cfg.USE_COGNITIVE_REGULARIZATION: self.entropix_engine = EntropixEngine(config=cfg, dtype=self.dtype, name="entropix_engine")
        self.output_head = nn.Dense(cfg.EFFECTIVE_VOCAB_SIZE, dtype=self.dtype)
    def __call__(self, tokens, deterministic=False):
        B, L = tokens.shape; x = self.token_embedding(tokens)
        cog_metrics, loss_components = None, None
        all_hidden_states: List[chex.Array] = [x]
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, deterministic)
            all_hidden_states.append(x)
            if self.config.USE_COGNITIVE_REGULARIZATION and i == len(self.transformer_blocks) - 1 and not deterministic:
                preliminary_logits = self.output_head(x)
                x, cog_metrics, loss_components = self.entropix_engine(x, preliminary_logits)
                all_hidden_states[-1] = x
        logits = self.output_head(x)
        if self.config.USE_COGNITIVE_REGULARIZATION and deterministic:
             cog_metrics = CognitiveMetrics(chaos_metric=jnp.zeros(B), kl_divergence=jnp.zeros(B))
             loss_components = LossComponents(loss_chaos=jnp.array(0.0), loss_divergence=jnp.array(0.0))
        return logits, cog_metrics, loss_components, all_hidden_states

def _create_pretrain_step(model: nn.Module, config: Config):
    @jit
    def pretrain_step_fn(state: CustomTrainState, super_batch: Dict[str, chex.Array], step_in_super_batch: int):
        dropout_key, new_base_key = jax.random.split(state.dropout_key)
        start_idx = step_in_super_batch * config.BATCH_SIZE
        batch = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start_idx, config.BATCH_SIZE, axis=0), super_batch)
        input_tokens, target_tokens, loss_mask = batch['input_ids'], batch['labels'], batch['loss_mask']
        B, H, W = input_tokens.shape
        input_tokens_flat, target_tokens_flat, loss_mask_flat = input_tokens.reshape(B, H*W), target_tokens.reshape(B, H*W), loss_mask.reshape(B, H*W)
        def loss_fn(params):
            logits_flat, cog_metrics, loss_components, all_hidden_states = model.apply({'params': params}, input_tokens_flat, rngs={'dropout': dropout_key})
            stable_logits = logits_flat.astype(jnp.float32)
            raw_loss = optax.softmax_cross_entropy_with_integer_labels(stable_logits, target_tokens_flat)
            model_loss = jnp.where(loss_mask_flat, raw_loss, 0).sum() / (loss_mask_flat.sum() + 1e-9)
            total_loss = model_loss
            metrics = {'loss': model_loss}
            if config.USE_COGNITIVE_REGULARIZATION:
                cog_loss = loss_components.loss_chaos + loss_components.loss_divergence
                total_loss += cog_loss * config.COGNITIVE_LOSS_WEIGHT
                metrics.update({'cognitive_loss': cog_loss, 'chaos_metric': cog_metrics.chaos_metric.mean(), 'kl_divergence': cog_metrics.kl_divergence.mean()})
            if config.USE_GEODESIC_REG:
                geodesic_losses = [jnp.mean(jnp.square(all_hidden_states[i+1] - all_hidden_states[i])) for i in range(len(all_hidden_states) - 1)]
                loss_geodesic = jnp.mean(jnp.array(geodesic_losses))
                total_loss += loss_geodesic * config.GEODESIC_LOSS_WEIGHT
                metrics['geodesic_loss'] = loss_geodesic
            if config.USE_COHERENCE_REG:
                final_hidden_state = all_hidden_states[-1]
                loss_coherence = jnp.mean(jnp.square(final_hidden_state[:, 1:] - final_hidden_state[:, :-1]))
                total_loss += loss_coherence * config.COHERENCE_LOSS_WEIGHT
                metrics['coherence_loss'] = loss_coherence
            metrics['total_loss'] = total_loss
            return total_loss, metrics
        (total_loss_val, metrics), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * config.EMA_DECAY + p * (1 - config.EMA_DECAY), state.ema_params, new_state.params)
        final_state = new_state.replace(ema_params=new_ema_params, dropout_key=new_base_key)
        metrics['learning_rate'] = config.LEARNING_RATE; metrics['grad_norm'] = optax.global_norm(grads)
        return final_state, metrics
    return pretrain_step_fn

class SotaTrainerJAX:
    def __init__(self, config: Config):
        self.config = config; self.console = Console(); self.interactive_state = InteractivityState()
        self.ui_lock = threading.Lock(); self.last_metrics: Dict[str, Any] = {}; self.steps_per_sec: float = 0.0; self.live_preview_panel: Optional[Panel] = None
        self.dtype = jnp.bfloat16 if config.USE_BFLOAT16 and jax.devices('gpu') else jnp.float32
        self.param_count = 0; self.last_submitted_prompt = None; self.file_handle, self.mmap_obj = None, None
        if self.config.USE_CANVAS_TRAINING:
            self.console.print(f"--- 🖼️  Initializing Canvas I/O Mode ({self.config.CANVAS_WIDTH}x{self.config.CANVAS_HEIGHT} x {self.config.NUM_CANVAS_PAGES} Pages)... ---")
            raw_text_path = self._prepare_data()
            try:
                self.file_handle = open(raw_text_path, 'rb'); self.file_size = os.fstat(self.file_handle.fileno()).st_size
                self.mmap_obj = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
                self.raw_text_mmap_view = np.frombuffer(self.mmap_obj, dtype=np.uint8)
                self.console.print(f"--- ✅ Memory-mapped {self.file_size/1e9:.2f} GB of text. ---")
            except Exception as e: self.console.print(f"[bold red]FATAL: Failed to mmap file: {e}[/bold red]"); sys.exit(1)
            self.last_submitted_prompt = "A high-quality professional photograph of a red apple on a wooden table."
    def _cleanup(self):
        self.console.print("--- Cleaning up resources... ---")
        if self.mmap_obj: self.mmap_obj.close()
        if self.file_handle: self.file_handle.close()
        self.console.print("--- Cleanup complete. ---")
    def shutdown(self, signum=None, frame=None):
        if not self.interactive_state.shutdown_event.is_set():
            self.console.print("\n--- Shutdown signal received. Cleaning up... ---", style="bold yellow")
            self.interactive_state.set_shutdown(); self._cleanup()
    def _word_wrap_tokens(self, tokens: np.ndarray, width: int) -> np.ndarray:
        wrapped_lines=[]; current_line=[]; words=np.split(tokens, np.where(tokens == 32)[0] + 1)
        for word in words:
            if not len(word): continue
            if len(current_line)+len(word)>width:
                wrapped_lines.extend(current_line); wrapped_lines.append(10)
                if len(word)>width:
                    current_line=list(word)
                    while len(current_line)>width: wrapped_lines.extend(current_line[:width]); wrapped_lines.append(10); current_line=current_line[width:]
                else: current_line=list(word)
            else: current_line.extend(list(word))
        wrapped_lines.extend(current_line)
        return np.array(wrapped_lines,dtype=np.int32)
    def _find_subsequence_np(self, haystack: np.ndarray, needle: np.ndarray) -> int:
        if len(needle) == 0: return 0
        if len(haystack) < len(needle): return -1
        possible_starts = np.where(haystack == needle[0])[0]
        for start_pos in possible_starts:
            if start_pos + len(needle) > len(haystack): continue
            if np.array_equal(haystack[start_pos : start_pos + len(needle)], needle): return start_pos
        return -1
    def _get_super_batch_from_mmap(self, num_examples):
        cfg = self.config; book_size = cfg.NUM_CANVAS_PAGES*cfg.CANVAS_HEIGHT*cfg.CANVAS_WIDTH
        batch_input_canvases=[]; batch_labels_canvases=[]; batch_loss_masks=[]
        user_marker=b"User:"; assistant_marker=b"Assistant:"
        user_marker_np=np.frombuffer(user_marker,dtype=np.uint8); assistant_marker_np=np.frombuffer(assistant_marker,dtype=np.uint8)
        for _ in range(num_examples):
            canvases=np.full((cfg.NUM_CANVAS_PAGES,cfg.CANVAS_HEIGHT,cfg.CANVAS_WIDTH),cfg.VOID_TOKEN_ID,dtype=np.int32)
            start_idx=np.random.randint(0,self.file_size-book_size*2); text_chunk_bytes=self.raw_text_mmap_view[start_idx:start_idx+book_size*2]
            page_idx,row,col=0,0,0; is_assistant_turn=False; last_pos=0
            while page_idx < cfg.NUM_CANVAS_PAGES and last_pos < len(text_chunk_bytes):
                chunk_to_search=text_chunk_bytes[last_pos:]; next_user_pos=self._find_subsequence_np(chunk_to_search,user_marker_np); next_asst_pos=self._find_subsequence_np(chunk_to_search,assistant_marker_np)
                if next_user_pos!=-1 and (next_asst_pos==-1 or next_user_pos<next_asst_pos): content_start=last_pos+next_user_pos+len(user_marker); content_end=last_pos+next_asst_pos if next_asst_pos!=-1 else len(text_chunk_bytes); is_assistant_turn=False; prefix=b"User: "
                elif next_asst_pos!=-1: content_start=last_pos+next_asst_pos+len(assistant_marker); content_end=last_pos+next_user_pos if next_user_pos!=-1 else len(text_chunk_bytes); is_assistant_turn=True; prefix=b"Assistant: "
                else: break
                content_bytes=text_chunk_bytes[content_start:content_end]; content_tokens=np.frombuffer(content_bytes,dtype=np.uint8).astype(np.int32); prefix_tokens=np.frombuffer(prefix,dtype=np.uint8).astype(np.int32)
                page_info_str=f"[P:{page_idx+1}] "; page_info_tokens=np.frombuffer(page_info_str.encode(),dtype=np.uint8).astype(np.int32)
                if row==0 and col==0: canvases[page_idx,0,:len(page_info_tokens)]=page_info_tokens; col=len(page_info_tokens)
                if col+len(prefix_tokens)<cfg.CANVAS_WIDTH: canvases[page_idx,row,col:col+len(prefix_tokens)]=prefix_tokens; col+=len(prefix_tokens)
                wrapped_tokens=self._word_wrap_tokens(content_tokens,cfg.CANVAS_WIDTH); current_word_start_col=col
                for token in wrapped_tokens:
                    if page_idx>=cfg.NUM_CANVAS_PAGES: break
                    if token==10: row+=1; col=0; current_word_start_col=0
                    else: canvases[page_idx,row,col]=token; col+=1
                    if col>=cfg.CANVAS_WIDTH: row+=1; col=0; current_word_start_col=0
                    if row>=cfg.CANVAS_HEIGHT:
                        canvases[page_idx,-1,-1]=cfg.CONTINUE_FWD_TOKEN_ID; page_idx+=1
                        if page_idx>=cfg.NUM_CANVAS_PAGES: break
                        row,col=0,0; page_info_str=f"[P:{page_idx+1}] "; page_info_tokens=np.frombuffer(page_info_str.encode(),dtype=np.uint8).astype(np.int32)
                        canvases[page_idx,0,:len(page_info_tokens)]=page_info_tokens; canvases[page_idx,0,len(page_info_tokens)]=cfg.CONTINUE_BWD_TOKEN_ID; col=len(page_info_tokens)+1; current_word_start_col=col
                if page_idx>=cfg.NUM_CANVAS_PAGES: break
                if not is_assistant_turn:
                    row+=1; col=0
                    if row<cfg.CANVAS_HEIGHT: canvases[page_idx,row,:]=ord('-')
                    row+=1; col=0
                    if row>=cfg.CANVAS_HEIGHT:
                        canvases[page_idx,-1,-1]=cfg.CONTINUE_FWD_TOKEN_ID; page_idx+=1
                        if page_idx>=cfg.NUM_CANVAS_PAGES: break
                        row,col=0,0
                if page_idx>=cfg.NUM_CANVAS_PAGES: break
                last_pos=content_end
            labels_book=canvases.copy(); mask_start_marker_np=np.frombuffer(b"Assistant:",dtype=np.uint8).astype(np.int32)
            loss_mask_book=np.zeros_like(canvases,dtype=bool); is_masking=False
            for p in range(canvases.shape[0]):
                for r in range(canvases.shape[1]):
                    start_idx=self._find_subsequence_np(canvases[p,r,:],mask_start_marker_np)
                    if start_idx!=-1:
                        is_masking=True; start_col=start_idx+len(mask_start_marker_np)
                        loss_mask_book[p,r,start_col:]=True
                        if r<canvases.shape[1]-1: loss_mask_book[p,r+1:,:]=True
                        if p<canvases.shape[0]-1: loss_mask_book[p+1:,:,:]=True
                        break
                if is_masking: break
            input_book=np.where(loss_mask_book,cfg.MASK_TOKEN_ID,labels_book); batch_input_canvases.extend(list(input_book)); batch_labels_canvases.extend(list(labels_book)); batch_loss_masks.extend(list(loss_mask_book))
        return {'input_ids':np.array(batch_input_canvases),'labels':np.array(batch_labels_canvases),'loss_mask':np.array(batch_loss_masks)}
    def _prepare_data(self):
        data_dir, raw_path = Path(self.config.DATA_DIR), Path(self.config.DATA_DIR) / self.config.RAW_TEXT_FILE
        data_dir.mkdir(exist_ok=True)
        if not raw_path.exists(): self.console.print(f"❌ [bold red]FATAL: Data file not found at {raw_path}.[/bold red]"); sys.exit(1)
        self.console.print(f"✅ Raw text data found: [cyan]{raw_path}[/cyan]", style="green"); return raw_path
    def _generate_layout(self, progress: Progress, global_step: int) -> Layout:
        with self.ui_lock:
            layout = Layout(); layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(name="footer", size=3))
            layout["main"].split_row(Layout(name="left", minimum_size=40, ratio=1), Layout(name="right", ratio=2))
            def to_float(value, default=0.0):
                try: return float(value)
                except (ValueError, TypeError): return default
            precision = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"🚀🧠 [bold]Ascended Thinker v3[/] | Step: {global_step} | SPS: {self.steps_per_sec:.2f} | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"
            layout["header"].update(Panel(Align.center(header_text), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))
            mem, util = self._get_gpu_stats()
            stats_tbl = Table.grid(expand=True, padding=(0, 1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:6.2f}[/] 🚀"); stats_tbl.add_row("Learning Rate", f"[green]{to_float(self.last_metrics.get('learning_rate')):.2e}[/]"); stats_tbl.add_row("GPU Mem/Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            loss_tbl = Table.grid(expand=True, padding=(0,1)); loss_tbl.add_column(style="dim"); loss_tbl.add_column(justify="right", style="bright_white")
            loss_tbl.add_row("Model Loss", f"{to_float(self.last_metrics.get('loss')):7.4f}")
            if self.config.USE_COGNITIVE_REGULARIZATION: loss_tbl.add_row(f"Cognitive Loss", f"{to_float(self.last_metrics.get('cognitive_loss')):7.4f}")
            if self.config.USE_GEODESIC_REG: loss_tbl.add_row(f"Geodesic Loss", f"{to_float(self.last_metrics.get('geodesic_loss')):7.4f}")
            if self.config.USE_COHERENCE_REG: loss_tbl.add_row(f"Coherence Loss", f"{to_float(self.last_metrics.get('coherence_loss')):7.4f}")
            loss_tbl.add_row("[bold]Total Loss[/]", f"[bold]{to_float(self.last_metrics.get('total_loss')):7.4f}[/]")
            left_panel_group_items = [Panel(stats_tbl, title="[bold]📊 Core Stats[/]"), Panel(loss_tbl, title="[bold]⚖️ Loss Components[/]", border_style="yellow")]
            if self.config.USE_COGNITIVE_REGULARIZATION:
                metrics_panel_content = Table.grid(expand=True, padding=(0,1)); metrics_panel_content.add_column(); metrics_panel_content.add_column(justify="right")
                metrics_panel_content.add_row("Chaos Metric", f"{to_float(self.last_metrics.get('chaos_metric')):7.4f}"); metrics_panel_content.add_row("KL Divergence", f"{to_float(self.last_metrics.get('kl_divergence')):7.4f}")
                left_panel_group_items.append(Panel(metrics_panel_content, title="[bold]🧠 Cognitive Metrics[/]", border_style="cyan"))
            layout["left"].update(Align.center(Group(*left_panel_group_items)))
            preview_panel = self.live_preview_panel
            if preview_panel: right_content = preview_panel
            else:
                placeholder_text = Align.center(Text("... training started, generation preview will appear periodically ...", justify="center"), vertical="middle")
                right_content = Panel(placeholder_text, height=self.config.CANVAS_HEIGHT + 2, border_style="dim")
            layout["right"].update(right_content)
            with self.interactive_state.lock:
                if self.interactive_state.is_prompting:
                    prompt_content = Panel(Text(self.interactive_state.current_prompt_text + "█", justify="left"), title="[bold yellow]Enter prompt (Enter to submit)[/]", border_style="yellow")
                    layout["footer"].update(prompt_content)
                else: layout["footer"].update(Padding(progress, (1,0)))
            return layout
    def _get_gpu_stats(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0); mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle); util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return f"{mem_info.used / 1024**3:.2f}/{mem_info.total / 1024**3:.2f} GiB", f"{util_info.gpu}%"
        except Exception: return "N/A", "N/A"
    @staticmethod
    def _format_canvas_to_panel(canvas_view: np.ndarray, title: str, config: Config) -> Panel:
        sub_canvas = canvas_view[:config.CANVAS_HEIGHT, :config.CANVAS_WIDTH]
        char_canvas = np.full(sub_canvas.shape, ' ', dtype='<U1')
        printable_mask = (sub_canvas >= 32) & (sub_canvas <= 126)
        char_canvas[printable_mask] = sub_canvas[printable_mask].astype(np.uint8).view('S1').astype(str)
        char_canvas[sub_canvas == config.VOID_TOKEN_ID] = config.VOID_CHAR; char_canvas[sub_canvas == config.MASK_TOKEN_ID] = config.MASK_CHAR
        char_canvas[sub_canvas == config.EXPAND_TOKEN_ID] = config.EXPAND_CHAR; char_canvas[sub_canvas == config.CONTINUE_FWD_TOKEN_ID] = config.CONTINUE_FWD_CHAR
        char_canvas[sub_canvas == config.CONTINUE_BWD_TOKEN_ID] = config.CONTINUE_BWD_CHAR; char_canvas[sub_canvas == config.SEMANTIC_CHECKSUM_TOKEN_ID] = config.SEMANTIC_CHECKSUM_CHAR
        full_string = '\n'.join([''.join(row) for row in char_canvas])
        lines = full_string.split('\n'); text_objects = []
        for line in lines:
            if "User:" in line: text_objects.append(Text(line, style="yellow bold"))
            elif "Assistant:" in line: text_objects.append(Text(line, style="green"))
            elif line.strip().startswith("----"): text_objects.append(Text(line, style="dim"))
            else: text_objects.append(Text(line, style="bright_white"))
            text_objects.append(Text("\n"))
        return Panel(Text.assemble(*text_objects), title=title, border_style="magenta", expand=False)
    @staticmethod
    @partial(jit, static_argnames=('apply_fn', 'config'))
    def _perform_sampled_generation(params, key, x_start, apply_fn, config: Config):
        B, H, W = x_start.shape; x_start_flat = x_start.reshape(B, H * W)
        logits_flat, _, _, _ = apply_fn({'params': params}, x_start_flat, deterministic=True, rngs={'dropout': key})
        special_tokens = jnp.array([config.VOID_TOKEN_ID, config.MASK_TOKEN_ID, config.EXPAND_TOKEN_ID, config.CONTINUE_FWD_TOKEN_ID, config.CONTINUE_BWD_TOKEN_ID, config.SEMANTIC_CHECKSUM_TOKEN_ID])
        special_token_mask = jnp.isin(jnp.arange(config.EFFECTIVE_VOCAB_SIZE), special_tokens)
        logits = jnp.where(special_token_mask, -jnp.inf, logits_flat) / config.TEMPERATURE
        _, top_k_indices = jax.lax.top_k(logits, k=config.TOP_K)
        top_k_mask = jnp.zeros_like(logits, dtype=jnp.bool_).at[jnp.arange(B)[:, None, None], jnp.arange(H * W)[None, :, None], top_k_indices].set(True)
        filtered_logits = jnp.where(top_k_mask, logits, -jnp.inf)
        samples_flat = jax.random.categorical(key, filtered_logits, axis=-1)
        is_mask_area_flat = (x_start_flat == config.MASK_TOKEN_ID)
        result_flat = jnp.where(is_mask_area_flat, samples_flat, x_start_flat)
        return result_flat.reshape(B, H, W)
    def _run_single_step_generation_preview(self, ema_params: dict, prompt_text: str, key: chex.PRNGKey) -> Panel:
        cfg = self.config; model = MaskedDenoisingModel(cfg, self.dtype)
        x_start = np.full((1, cfg.CANVAS_HEIGHT, cfg.CANVAS_WIDTH), cfg.VOID_TOKEN_ID, dtype=np.int32)
        user_prefix = np.frombuffer(b"User: ", dtype=np.uint8).astype(np.int32); prompt_tokens = np.frombuffer(prompt_text.encode('utf-8', 'replace'), dtype=np.uint8).astype(np.int32)
        full_prompt_tokens = np.concatenate([user_prefix, prompt_tokens]); wrapped_prompt = self._word_wrap_tokens(full_prompt_tokens, cfg.CANVAS_WIDTH)
        row, col = 0, 0
        for token in wrapped_prompt:
            if row >= cfg.CANVAS_HEIGHT: break
            if token == 10: row += 1; col = 0; continue
            x_start[0, row, col] = token; col += 1
            if col >= cfg.CANVAS_WIDTH: row += 1; col = 0
        row += 1
        if row < cfg.CANVAS_HEIGHT: x_start[0, row, :] = ord('-')
        row += 1
        if row < cfg.CANVAS_HEIGHT: asst_prefix = np.frombuffer(b"Assistant: ", dtype=np.uint8).astype(np.int32); x_start[0, row, :len(asst_prefix)] = asst_prefix
        mask_start_row = row; mask_start_col = len(asst_prefix) if row < cfg.CANVAS_HEIGHT else 0
        if mask_start_row < cfg.CANVAS_HEIGHT:
            x_start[0, mask_start_row, mask_start_col:] = cfg.MASK_TOKEN_ID
            if mask_start_row + 1 < cfg.CANVAS_HEIGHT: x_start[0, mask_start_row + 1:, :] = cfg.MASK_TOKEN_ID
        x_start_jax = jnp.array(x_start)
        fully_unmasked = self._perform_sampled_generation(ema_params, key, x_start_jax, model.apply, cfg)
        fully_unmasked.block_until_ready()
        final_canvas_view = np.array(fully_unmasked).squeeze()
        final_title = f"🖼️ Canvas Generation (T={cfg.TEMPERATURE}, K={cfg.TOP_K}, Step {key[1]})"
        return self._format_canvas_to_panel(final_canvas_view, final_title, cfg)
    def _save_checkpoint(self, state, path):
        path.parent.mkdir(exist_ok=True); state_cpu = jax.device_get(state)
        state_cpu = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16 else x, state_cpu)
        path.write_bytes(serialization.to_bytes(state_cpu))
        self.console.print(f"\n--- 💾 Checkpoint saved to [cyan]{path}[/cyan] (step {int(state.step)}) ---")
    def train(self):
        signal.signal(signal.SIGINT, self.shutdown); signal.signal(signal.SIGTERM, self.shutdown)
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        model = MaskedDenoisingModel(self.config, self.dtype)
        key = jax.random.PRNGKey(42); main_key, params_key, dropout_key = jax.random.split(key, 3)
        dummy_tokens = jnp.zeros((self.config.BATCH_SIZE, self.config.CANVAS_HEIGHT * self.config.CANVAS_WIDTH), dtype=jnp.int32)
        params = model.init({'params': params_key, 'dropout': dropout_key}, dummy_tokens, deterministic=True)['params']
        self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        self.console.print(f'--- 🚀 [bold]Ascended Thinker v3[/bold] initialized with [yellow]{self.param_count:,}[/yellow] parameters. ---', style="magenta")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), toroidal_gradient_transform(), optax.adamw(learning_rate=self.config.LEARNING_RATE, b1=0.9, b2=0.95))
        state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=optimizer, ema_params=params, dropout_key=dropout_key)
        ckpt_path = Path(self.config.CHECKPOINT_DIR) / f"{self.config.BASENAME}_canvas.ckpt"
        if not self.config.FRESH_START and ckpt_path.exists():
            self.console.print(f"--- Resuming from checkpoint: [cyan]{ckpt_path}[/cyan] ---");
            with ckpt_path.open('rb') as f: state = serialization.from_bytes(state, f.read())
            self.console.print(f"--- Resumed from step {int(state.step)}. ---")
        jitted_train_step = _create_pretrain_step(model, self.config)
        self.console.print("--- Compiling JAX functions & pre-loading data... ---")
        device_batch_size = self.config.SUPER_BATCH_SIZE * self.config.NUM_CANVAS_PAGES
        super_batch_host = self._get_super_batch_from_mmap(self.config.SUPER_BATCH_SIZE)
        super_batch_device = jax.device_put(super_batch_host)
        state, _ = jitted_train_step(state, super_batch_device, 0)
        self.console.print("--- ✅ Compilation complete. Starting training. ---")
        step_in_super_batch = 0; global_step = int(state.step)
        progress = Progress(TextColumn("[bold]Training Progress[/]"), BarColumn(), "•", TextColumn("Step {task.completed}"), TimeRemainingColumn())
        main_task = progress.add_task("steps", total=self.config.EPOCHS * (self.file_size // (self.config.CANVAS_HEIGHT * self.config.CANVAS_WIDTH)), completed=global_step)
        live = Live(self._generate_layout(progress, global_step), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        try:
            live.start()
            with ThreadPoolExecutor(max_workers=1) as gen_pool, ThreadPoolExecutor(max_workers=1) as data_loader_pool:
                active_generation_future = None
                next_super_batch_future = data_loader_pool.submit(self._get_super_batch_from_mmap, self.config.SUPER_BATCH_SIZE)
                while not self.interactive_state.shutdown_event.is_set():
                    last_step_time = time.time()
                    state, metrics = jitted_train_step(state, super_batch_device, step_in_super_batch)
                    self.steps_per_sec = 1.0 / (time.time() - last_step_time + 1e-9)
                    self.last_metrics = jax.device_get(metrics)
                    global_step += 1; step_in_super_batch += 1
                    progress.update(main_task, completed=global_step)
                    if step_in_super_batch >= device_batch_size // self.config.BATCH_SIZE:
                        super_batch_device = jax.device_put(next_super_batch_future.result())
                        next_super_batch_future = data_loader_pool.submit(self._get_super_batch_from_mmap, self.config.SUPER_BATCH_SIZE)
                        step_in_super_batch = 0
                    
                    submitted_prompt = self.interactive_state.get_submitted_prompt()
                    if submitted_prompt: self.last_submitted_prompt = submitted_prompt
                    
                    if active_generation_future and active_generation_future.done():
                        with self.ui_lock: self.live_preview_panel = active_generation_future.result()
                        active_generation_future = None

                    trigger_by_step = (global_step % self.config.PREVIEW_EVERY_N_STEPS == 0)
                    if (submitted_prompt or trigger_by_step) and self.last_submitted_prompt and not active_generation_future:
                        ema_params_cpu = jax.device_get(state.ema_params)
                        preview_key = jax.random.PRNGKey(global_step) 
                        active_generation_future = gen_pool.submit(self._run_single_step_generation_preview, ema_params_cpu, self.last_submitted_prompt, preview_key)

                    if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and global_step % self.config.SAVE_EVERY == 0):
                        self._save_checkpoint(state, ckpt_path)
                    
                    live.update(self._generate_layout(progress, global_step), refresh=True)

        finally:
            self.shutdown(); live.stop()
            self.console.print("\n--- Training loop terminated. Saving final state... ---")
            if 'state' in locals(): self._save_checkpoint(state, ckpt_path)
            key_listener_thread.join(timeout=1)

def main():
    parser = argparse.ArgumentParser(description="Ascended Thinker v3 (JAX): Physics-Enhanced Canvas Trainer")
    cfg = Config()
    parser.add_argument('--data-dir', type=str, default=cfg.DATA_DIR); parser.add_argument('--raw-text-file', type=str, default=cfg.RAW_TEXT_FILE)
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS); parser.add_argument('--learning-rate', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--disable-cognitive', action='store_true', help="Disable the Entropix cognitive engine.")
    parser.add_argument('--disable-grammar-field', action='store_true', help="Disable the Grammar Field Operator.")
    parser.add_argument('--disable-geodesic-reg', action='store_true', help="Disable Geodesic (smooth path) regularization.")
    parser.add_argument('--disable-coherence-reg', action='store_true', help="Disable Coherence (local similarity) regularization.")
    args = parser.parse_args()

    if args.disable_cognitive: cfg.USE_COGNITIVE_REGULARIZATION = False
    if args.disable_grammar_field: cfg.USE_GRAMMAR_FIELD = False
    if args.disable_geodesic_reg: cfg.USE_GEODESIC_REG = False
    if args.disable_coherence_reg: cfg.USE_COHERENCE_REG = False
    if args.fresh_start: cfg.FRESH_START = True
    for arg, value in vars(args).items():
        if hasattr(cfg, arg.upper()): setattr(cfg, arg.upper(), value)
    SotaTrainerJAX(cfg).train()

if __name__ == "__main__":
    main()