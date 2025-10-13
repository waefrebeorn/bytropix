import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
except NameError: 
    cache_dir = os.path.join(os.path.expanduser("~"), ".jax_cache_ascended")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
import sys
import time
from pathlib import Path
import math
import pickle
import signal
import threading
import platform
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
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
                ("rich.console", "rich"), ("pynvml", "pynvml")]
missing = [pkg for mod, pkg in dependencies if not __import__(mod.split('.')[0], fromlist=['_'] * (mod.count('.') + 1))]
if missing:
    print(f"\n[FATAL] Missing dependencies. Please run: pip install {' '.join(missing)}")
    sys.exit(1)
print("--- All dependencies verified. ---")
import tensorflow as tf
import tensorflow_datasets as tfds
from rich.live import Live; from rich.table import Table; from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
from rich.text import Text
import pynvml; pynvml.nvmlInit()
tf.config.set_visible_devices([], 'GPU')
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
class Config:
    DATA_DIR = "./data"; RAW_TEXT_FILE = "open_orca_formatted.txt"; BASENAME = "AscendedThinker_v2_Jax"; CHECKPOINT_DIR = "./checkpoints"
    VOCAB_SIZE = 256; D_MODEL = 512; NUM_LAYERS = 2; NUM_HEADS = 64; D_FF = D_MODEL * 8; DROPOUT = 0.1
    NUM_GROUPS = 4
    EMA_DECAY = 0.999
    BLOCK_SIZE = 512; CHUNK_SIZE = 32; 
    N_SUPERVISION_STEPS = 4; N_LATENT_RECURSION_STEPS = 4
    EPOCHS = 300; BATCH_SIZE = 1; REBATCH_SIZE = 1; LEARNING_RATE = 5e-5; USE_BFLOAT16 = True; SAVE_EVERY = 2000; FRESH_START = False
    USE_Q_CONTROLLER = True; Q_WARMUP_STEPS = 20
    ENTROPIX_ENABLED = True; USE_COGNITIVE_LOSS = True
    COGNITIVE_LOSS_WARMUP_STEPS = 20; INITIAL_COGNITIVE_LOSS_WEIGHT = 0.01
    USE_PID_CONTROLLERS = True; PID_GAINS_CHAOS = (0.5, 0.005, 0.2); PID_GAINS_DIVERGENCE = (0.8, 0.01, 0.3); PID_TARGET_CHAOS_METRIC = 0.1; PID_TARGET_DIVERGENCE_METRIC = 0.5
    PID_WEIGHT_LIMIT = 1.0
    DIRICHLET_SUPPORT_SIZE = 256; DIRICHLET_SUPPORT_PATH = "./dirichlet_support.npy"; EMWA_ENT_NAKED_COEFF = 0.99; EMWA_VARENT_NAKED_COEFF = 0.99; EMWA_LOGP_UPDATE_COEFF = 0.05
    THRESHOLD_LELV = (0.3, 1.2); THRESHOLD_HELV = (2.5, 1.2); THRESHOLD_LEHV = (1.2, 2.5); THRESHOLD_HEHV = (2.5, 2.5); CLARIFY_TOKEN_ID = 63
class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock(); self.preview_index_change = 0
        self.shutdown_event, self.force_save, self.force_preview = threading.Event(), False, False
    def get_and_reset_preview_change(self) -> int:
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self) -> bool:
        with self.lock: save = self.force_save; self.force_save = False; return save
    def get_and_reset_force_preview(self) -> bool:
        with self.lock: preview = self.force_preview; self.force_preview = False; return preview
    def set_shutdown(self): self.shutdown_event.set()
def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [g] Force Preview | [s] Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt
    else: import select, sys, tty, termios
    try:
        if platform.system() != "Windows":
            fd, old_settings = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        while not shared_state.shutdown_event.is_set():
            has_input = (platform.system() == "Windows" and msvcrt.kbhit()) or (platform.system() != "Windows" and select.select([sys.stdin], [], [], 0.05)[0])
            if has_input:
                key = msvcrt.getch().decode() if platform.system() == "Windows" else sys.stdin.read(1)
                if key in ['q', '\x03']: shared_state.set_shutdown(); break
                elif key == 's':
                    with shared_state.lock: shared_state.force_save = True
                elif key == 'g':
                    with shared_state.lock: shared_state.force_preview = True
            time.sleep(0.05)
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
@struct.dataclass
class DSState:
    emwa_logp_on_supp: chex.Array; emwa_ent_naked: chex.Array; emwa_varent_naked: chex.Array
@struct.dataclass
class CognitiveMetrics:
    naked_entropy: chex.Array; naked_varentropy: chex.Array; kl_divergence: chex.Array; chaos_metric: chex.Array
@struct.dataclass
class LossComponents:
    loss_chaos: chex.Array; loss_divergence: chex.Array
@struct.dataclass
class PIDState:
    integral: float; prev_error: float; weight: float
def pid_update(state: PIDState, current_metric_value: float, target: float, gains: Tuple[float, float, float], weight_limit: float, dt: float = 1.0) -> Tuple[PIDState, float]:
    kp, ki, kd = gains
    error = current_metric_value - target
    integral = state.integral + error * dt
    derivative = (error - state.prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    new_weight = jnp.clip(state.weight + output, 0.0, weight_limit)
    new_state = PIDState(integral=integral, prev_error=error, weight=new_weight)
    return new_state, new_weight
Q_CONTROLLER_CONFIG_TEXT = {
    "q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1],
    "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 3e-4,
    "metric_history_len": 200, "loss_min": 0.01, "loss_max": 6.0, "exploration_rate_q": 0.5,
    "min_exploration_rate": 0.05, "exploration_decay": 0.9998, "trend_window": 10,
    "improve_threshold": 1e-4, "regress_threshold": 1e-5, "regress_penalty": -10.0,
    "stagnation_penalty": -2.0, "warmup_steps": Config.Q_WARMUP_STEPS, "warmup_lr_start": Config.LEARNING_RATE / 100
}
@struct.dataclass
class QControllerState:
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray; exploration_rate: jnp.ndarray
    step_count: jnp.ndarray; last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray
def init_q_controller(config):
    return QControllerState(
        q_table=jnp.zeros((config["q_table_size"], config["num_lr_actions"])),
        metric_history=jnp.zeros((config["metric_history_len"],)),
        current_lr=jnp.array(config["warmup_lr_start"], dtype=jnp.float32),
        exploration_rate=jnp.array(config["exploration_rate_q"]),
        step_count=jnp.array(0, dtype=jnp.int32),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0),
        status_code=jnp.array(0, dtype=jnp.int32)
    )
@jit
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey, target_lr: float):
    config = Q_CONTROLLER_CONFIG_TEXT
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]
        new_lr = config["warmup_lr_start"] * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - config['trend_window'], config['trend_window']))
        state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        explore_key, action_key = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate,
            lambda: jax.random.randint(action_key, (), 0, config["num_lr_actions"]),
            lambda: jnp.argmax(state.q_table[state_idx]))
        new_lr = jnp.clip(state.current_lr * jnp.array(config["lr_change_factors"])[action_idx], config["lr_min"], config["lr_max"])
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)
@jit
def q_controller_update(state: QControllerState, metric_value: float):
    config = Q_CONTROLLER_CONFIG_TEXT
    metric_value_f32 = metric_value.astype(jnp.float32)
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value_f32)
    def perform_update(st):
        y = jax.lax.dynamic_slice_in_dim(new_metric_history, config["metric_history_len"] - config['trend_window'], config['trend_window'])
        x = jnp.arange(config["trend_window"], dtype=jnp.float32)
        A = jnp.vstack([x, jnp.ones_like(x)]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(slope < -config["improve_threshold"],
            lambda: (jnp.array(1), jnp.abs(slope) * 10.0),
            lambda: jax.lax.cond(slope > config["regress_threshold"],
                lambda: (jnp.array(3), -jnp.abs(slope) * 10.0 - config["regress_penalty"]),
                lambda: (jnp.array(2), config["stagnation_penalty"])))
        last_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config["metric_history_len"] - config['trend_window'] -1, config['trend_window']))
        last_state_idx = jnp.clip(((last_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        new_mean = jnp.mean(y)
        next_state_idx = jnp.clip(((new_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        current_q = st.q_table[last_state_idx, st.last_action_idx]
        max_next_q = jnp.max(st.q_table[next_state_idx])
        new_q = current_q + config["learning_rate_q"] * (reward + config["discount_factor_q"] * max_next_q - current_q)
        new_q_table = st.q_table.at[last_state_idx, st.last_action_idx].set(new_q.astype(st.q_table.dtype))
        new_exp_rate = jnp.maximum(config["min_exploration_rate"], st.exploration_rate * config["exploration_decay"])
        return st.replace(q_table=new_q_table, exploration_rate=new_exp_rate, last_reward=reward.astype(st.last_reward.dtype), status_code=status_code)
    can_update = (state.step_count > config["warmup_steps"]) & (state.last_action_idx >= 0)
    new_state = jax.lax.cond(can_update, perform_update, lambda s: s, state)
    return new_state.replace(metric_history=new_metric_history)
class JaxTrainState(train_state.TrainState):
    ema_params: Any
    q_state: QControllerState
    pid_chaos_state: PIDState
    pid_divergence_state: PIDState
    dropout_key: chex.PRNGKey
class EntropixEngine(nn.Module):
    config: Config
    dirichlet_support: chex.Array
    def _normalize_logits(self, logits: chex.Array) -> chex.Array:
        return nn.log_softmax(logits, axis=-1)
    def _ent_varent(self, log_probs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        probs = jnp.exp(log_probs)
        ent = -jnp.sum(probs * log_probs, axis=-1)
        diff = log_probs + jnp.expand_dims(ent, -1)
        varent = jnp.sum(probs * jnp.power(diff, 2), axis=-1)
        return ent, varent
    def _kl_divergence(self, logp: chex.Array, logq: chex.Array) -> chex.Array:
        p = jnp.exp(logp)
        return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)
    def initialize_state(self, bsz: int, dtype: Any) -> DSState:
        supp_size = self.config.DIRICHLET_SUPPORT_SIZE
        return DSState(
            emwa_logp_on_supp=jnp.full((bsz, supp_size), -math.log(supp_size), dtype=dtype),
            emwa_ent_naked=jnp.zeros(bsz, dtype=dtype),
            emwa_varent_naked=jnp.zeros(bsz, dtype=dtype)
        )
    @nn.compact
    def __call__(self, logits_on_supp: chex.Array, prev_state: DSState) -> Tuple[CognitiveMetrics, LossComponents, DSState]:
        naked_log_probs_on_supp = self._normalize_logits(logits_on_supp)
        naked_ent, naked_varent = self._ent_varent(naked_log_probs_on_supp)
        kl_div = self._kl_divergence(naked_log_probs_on_supp, prev_state.emwa_logp_on_supp)
        new_emwa_ent_naked = (self.config.EMWA_ENT_NAKED_COEFF * naked_ent + 
                              (1.0 - self.config.EMWA_ENT_NAKED_COEFF) * prev_state.emwa_ent_naked)
        new_emwa_varent_naked = (self.config.EMWA_VARENT_NAKED_COEFF * naked_varent +
                                 (1.0 - self.config.EMWA_VARENT_NAKED_COEFF) * prev_state.emwa_varent_naked)
        new_emwa_logp_on_supp = (self.config.EMWA_LOGP_UPDATE_COEFF * naked_log_probs_on_supp +
                                 (1.0 - self.config.EMWA_LOGP_UPDATE_COEFF) * prev_state.emwa_logp_on_supp)
        new_state = DSState(
            emwa_logp_on_supp=new_emwa_logp_on_supp,
            emwa_ent_naked=new_emwa_ent_naked,
            emwa_varent_naked=new_emwa_varent_naked,
        )
        metrics = CognitiveMetrics(
            naked_entropy=naked_ent, naked_varentropy=naked_varent, kl_divergence=kl_div,
            chaos_metric=(naked_ent * naked_varent)
        )
        loss_comp = LossComponents(
            loss_chaos=jnp.mean(metrics.chaos_metric),
            loss_divergence=jnp.mean(metrics.kl_divergence)
        )
        return metrics, loss_comp, new_state
    def sample(self, logits: chex.Array, state: DSState, key: chex.PRNGKey) -> Tuple[chex.Array, DSState]:
        log_probs = self._normalize_logits(logits)
        ent, varent = self._ent_varent(log_probs)
        is_helv = (ent > self.config.THRESHOLD_HELV[0]) & (varent < self.config.THRESHOLD_HELV[1])
        is_hehv = (ent > self.config.THRESHOLD_HEHV[0]) & (varent > self.config.THRESHOLD_HEHV[1])
        next_token_argmax = jnp.argmax(log_probs, axis=-1)
        probs_hehv = nn.softmax(logits, axis=-1)
        sampled_tokens_hehv = jax.random.categorical(key, logits)
        next_token = jnp.where(is_helv, self.config.CLARIFY_TOKEN_ID, next_token_argmax)
        next_token = jnp.where(is_hehv, sampled_tokens_hehv, next_token)
        logits_on_supp = jnp.take_along_axis(logits, jnp.expand_dims(self.dirichlet_support, 0), axis=-1)
        _, _, new_state = self.__call__(logits_on_supp, state)
        return next_token, new_state
class SingleGroupProcessor(nn.Module):
    num_heads_group: int; d_group: int; d_ff_group: int; dropout: float; dtype: Any
    @nn.compact
    def __call__(self, group_x, causal_mask=None, deterministic=False):
        norm_x = nn.LayerNorm(dtype=self.dtype)(group_x)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads_group, qkv_features=self.d_group,
            dropout_rate=self.dropout, dtype=self.dtype
        )(norm_x, norm_x, mask=causal_mask, deterministic=deterministic)
        group_x = group_x + attn_output
        norm_x = nn.LayerNorm(dtype=self.dtype)(group_x)
        ffn_output = nn.Dense(self.d_ff_group, dtype=self.dtype)(norm_x)
        ffn_output = nn.gelu(ffn_output)
        ffn_output = nn.Dense(self.d_group, dtype=self.dtype)(ffn_output)
        ffn_output = nn.Dropout(self.dropout)(ffn_output, deterministic=deterministic)
        group_x = group_x + ffn_output
        return group_x
class GroupedTransformerBlock(nn.Module):
    d_model: int; num_heads: int; d_ff: int; dropout: float; num_groups: int; dtype: Any
    def setup(self):
        d_group = self.d_model // self.num_groups
        num_heads_group = self.num_heads // self.num_groups
        d_ff_group = self.d_ff // self.num_groups
        self.vmapped_processor = nn.vmap(
            SingleGroupProcessor,
            variable_axes={'params': 0},
            split_rngs={'params': True, 'dropout': True},
            in_axes=(2, None, None),
            out_axes=2,
        )(
            num_heads_group=num_heads_group, d_group=d_group, d_ff_group=d_ff_group,
            dropout=self.dropout, dtype=self.dtype
        )
    def __call__(self, x, causal_mask=None, deterministic=False):
        batch_size, seq_len, _ = x.shape
        d_group = self.d_model // self.num_groups
        x_grouped = x.reshape(batch_size, seq_len, self.num_groups, d_group)
        processed_groups = self.vmapped_processor(x_grouped, causal_mask, deterministic)
        return processed_groups.reshape(batch_size, seq_len, self.d_model)
class GroupedReasoningBlock(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, x, mask, deterministic):
        for _ in range(self.config.NUM_LAYERS):
            x = GroupedTransformerBlock(
                d_model=self.config.D_MODEL, num_heads=self.config.NUM_HEADS,
                d_ff=self.config.D_FF, dropout=self.config.DROPOUT,
                num_groups=self.config.NUM_GROUPS, dtype=self.dtype
            )(x, mask, deterministic)
        return x
class TinyRecursiveMarkovianThinker(nn.Module):
    config: Config; dtype: Any
    @staticmethod
    def get_dirichlet_support(config: Config):
        support_path = Path(config.DIRICHLET_SUPPORT_PATH)
        if not support_path.exists():
            if config.DIRICHLET_SUPPORT_SIZE > config.VOCAB_SIZE:
                raise ValueError("DIRICHLET_SUPPORT_SIZE > VOCAB_SIZE")
            key = jax.random.PRNGKey(0) 
            support = jax.random.permutation(key, config.VOCAB_SIZE)[:config.DIRICHLET_SUPPORT_SIZE]
            np.save(support_path, np.array(support))
            return support
        return jnp.array(np.load(support_path))
    def setup(self):
        dirichlet_support = self.get_dirichlet_support(self.config)
        self.entropix_engine = EntropixEngine(self.config, dirichlet_support)
        self.token_embedding = nn.Embed(self.config.VOCAB_SIZE, self.config.D_MODEL, dtype=self.dtype)
        self.input_dropout = nn.Dropout(self.config.DROPOUT)
        self.reasoning_net = GroupedReasoningBlock(self.config, self.dtype)
        self.z_update_proj = nn.Dense(self.config.D_MODEL, dtype=self.dtype)
        self.y_update_proj = nn.Dense(self.config.D_MODEL, dtype=self.dtype)
        self.output_head = nn.Dense(self.config.VOCAB_SIZE, dtype=self.dtype)
        self.y_init = self.param('y_init', nn.initializers.normal(), (1, self.config.CHUNK_SIZE, self.config.D_MODEL))
        self.z_init = self.param('z_init', nn.initializers.normal(), (1, self.config.CHUNK_SIZE, self.config.D_MODEL))
        self.causal_mask = nn.make_causal_mask(jnp.ones((1, self.config.CHUNK_SIZE), dtype="bool"))
        position = jnp.arange(0, self.config.CHUNK_SIZE, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.config.D_MODEL, 2) * (-math.log(10000.0) / self.config.D_MODEL))
        pe = jnp.zeros((self.config.CHUNK_SIZE, self.config.D_MODEL))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pos_encoding = pe
    def __call__(self, indices, y_start, z_start, ds_state, global_step, deterministic=False) -> Tuple:
        batch_size = indices.shape[0]
        x_embed = self.token_embedding(indices) + self.pos_encoding
        x_embed = self.input_dropout(x_embed, deterministic=deterministic)
        y, z = y_start, z_start
        for _ in range(self.config.N_LATENT_RECURSION_STEPS):
            z_input_raw = jnp.concatenate([x_embed, y, z], axis=-1)
            z_input_proj = self.z_update_proj(z_input_raw)
            z = self.reasoning_net(z_input_proj, self.causal_mask, deterministic)
            y_input_raw = jnp.concatenate([y, z], axis=-1)
            y_input_proj = self.y_update_proj(y_input_raw)
            y = self.reasoning_net(y_input_proj, self.causal_mask, deterministic)
        y_final, z_final = y, z
        logits = self.output_head(y_final)
        is_cog_loss_active = self.config.ENTROPIX_ENABLED and self.config.USE_COGNITIVE_LOSS and global_step > self.config.COGNITIVE_LOSS_WARMUP_STEPS
        def compute_cog_loss():
            last_token_logits = logits[:, -1, :]
            logits_on_supp = jnp.take_along_axis(last_token_logits, jnp.expand_dims(self.entropix_engine.dirichlet_support, 0), axis=-1)
            cog_metrics, loss_comp, new_ds_state = self.entropix_engine(logits_on_supp, ds_state)
            return cog_metrics, loss_comp, new_ds_state
        def dummy_cog_loss():
            dtype = self.dtype
            batch_shape = (batch_size,)
            dummy_metrics = CognitiveMetrics(
                naked_entropy=jnp.zeros(batch_shape, dtype=dtype),
                naked_varentropy=jnp.zeros(batch_shape, dtype=dtype),
                kl_divergence=jnp.zeros(batch_shape, dtype=dtype),
                chaos_metric=jnp.zeros(batch_shape, dtype=dtype)
            )
            dummy_loss_comp = LossComponents(
                loss_chaos=jnp.array(0.0, dtype=dtype),
                loss_divergence=jnp.array(0.0, dtype=dtype)
            )
            return dummy_metrics, dummy_loss_comp, ds_state
        cog_metrics, loss_components, new_ds_state = jax.lax.cond(is_cog_loss_active, compute_cog_loss, dummy_cog_loss)
        return (y_final, z_final), logits, cog_metrics, loss_components, new_ds_state
    def sample_with_entropix(self, logits: chex.Array, ds_state: DSState, key: chex.PRNGKey) -> Tuple[chex.Array, DSState]:
        return self.entropix_engine.sample(logits, ds_state, key)
@partial(jit, static_argnames=('model_apply', 'config', 'max_new_tokens'))
def generate_uncompiled_jax(params, prompt_indices, key, model_apply, config: Config, max_new_tokens):
    prompt_len = prompt_indices.shape[1]
    total_len = prompt_len + max_new_tokens
    padded_prompt = jnp.pad(prompt_indices, ((0, 0), (0, total_len - prompt_len)), 'constant')
    y = jnp.expand_dims(params['y_init'][0], 0)
    z = jnp.expand_dims(params['z_init'][0], 0)
    support_np = np.load(config.DIRICHLET_SUPPORT_PATH)
    temp_entropix_engine = EntropixEngine(config, dirichlet_support=jnp.array(support_np))
    ds_state = temp_entropix_engine.initialize_state(1, y.dtype)
    def loop_body(i, carry):
        current_token_idx = prompt_len + i
        generated_indices, y, z, ds_state, key = carry
        context_start = jnp.maximum(0, current_token_idx - config.CHUNK_SIZE)
        input_chunk_indices = jax.lax.dynamic_slice_in_dim(generated_indices, context_start, config.CHUNK_SIZE, axis=1)
        (y_rec, z_rec), logits, _, _, _ = model_apply(
            {'params': params}, input_chunk_indices, y, z, ds_state, global_step=999999, deterministic=True
        )
        logits_pos = jnp.minimum(current_token_idx, config.CHUNK_SIZE - 1)
        logits_y = jax.lax.dynamic_slice_in_dim(logits, logits_pos, 1, axis=1)
        key, sample_key = jax.random.split(key)
        if config.ENTROPIX_ENABLED:
            next_token, ds_state_new = model_apply(
                {'params': params}, jnp.squeeze(logits_y, 1), ds_state, sample_key,
                method=TinyRecursiveMarkovianThinker.sample_with_entropix
            )
        else:
            next_token = jax.random.categorical(sample_key, jnp.squeeze(logits_y, 1))
            ds_state_new = ds_state
        update_token = next_token.reshape(1, 1)
        new_generated_indices = jax.lax.dynamic_update_slice(
            generated_indices, 
            update_token, 
            (0, current_token_idx)
        )
        return new_generated_indices, y_rec, z_rec, ds_state_new, key
    initial_carry = (padded_prompt, y, z, ds_state, key)
    final_indices, _, _, _, _ = jax.lax.fori_loop(0, max_new_tokens, loop_body, initial_carry)
    return final_indices.squeeze(0)[prompt_len:]
def create_direct_text_dataset(text_path: Path, block_size: int, batch_size: int, rebatch_size: int, is_training: bool):
    raw_text = tf.io.read_file(str(text_path))
    byte_tensor = tf.io.decode_raw(raw_text, tf.uint8)
    ds = tf.data.Dataset.from_tensor_slices(byte_tensor)
    ds = ds.window(block_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(block_size + 1))
    ds = ds.map(lambda chunk: (tf.cast(chunk[:-1], tf.int64), tf.cast(chunk[1:], tf.int64)), 
                num_parallel_calls=tf.data.AUTOTUNE)
    num_examples = len(byte_tensor) // (block_size + 1)
    if is_training:
        ds = ds.shuffle(buffer_size=10000).repeat()
    super_batch_size = batch_size * rebatch_size
    ds = ds.batch(super_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds, num_examples
class SotaTrainerJAX:
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.interactive_state = InteractivityState()
        self.dtype = jnp.bfloat16 if config.USE_BFLOAT16 and (jax.devices('gpu')[0].platform.upper() in ('CUDA', 'ROCM') if jax.devices('gpu') else False) else jnp.float32
        self.pid_controllers = {}
        if self.config.ENTROPIX_ENABLED and self.config.USE_PID_CONTROLLERS:
            self.pid_controllers['chaos'] = (self.config.PID_TARGET_CHAOS_METRIC, self.config.PID_GAINS_CHAOS)
            self.pid_controllers['divergence'] = (self.config.PID_TARGET_DIVERGENCE_METRIC, self.config.PID_GAINS_DIVERGENCE)
        self.param_count = 0; self.active_preview_future = None; self.async_pool = ThreadPoolExecutor(max_workers=1)
        self.preview_data_buffer, self.preview_index = [], 0; self.shared_state = {}
    def shutdown(self, signum=None, frame=None):
        if not self.interactive_state.shutdown_event.is_set():
            self.console.print("\n--- Shutdown signal received. Cleaning up... ---", style="bold yellow")
            self.interactive_state.set_shutdown()
            self.async_pool.shutdown(wait=False, cancel_futures=True)
    def _prepare_data(self):
        data_dir, raw_path = Path(self.config.DATA_DIR), Path(self.config.DATA_DIR) / self.config.RAW_TEXT_FILE
        data_dir.mkdir(exist_ok=True)
        if not raw_path.exists():
            self.console.print(f"❌ [bold red]FATAL: Data file not found at {raw_path}.[/bold red]"); sys.exit(1)
        self.console.print(f"✅ Raw text data found: [cyan]{raw_path}[/cyan]", style="green")
        return raw_path
    def _save_checkpoint(self, state: JaxTrainState, path: Path):
        state_cpu = jax.device_get(state)
        state_cpu = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16 else x, state_cpu)
        serialized_state = serialization.to_bytes(state_cpu)
        path.write_bytes(serialized_state)
        self.console.print(f"\n--- 💾 Checkpoint saved to [cyan]{path}[/cyan] (step {int(state.step)}) ---", style="blue")
    def _run_live_generation_preview(self, ema_params: dict, prompt_text: str, key: chex.PRNGKey):
        prompt_bytes = jnp.array([list(prompt_text.encode('utf-8'))], dtype=jnp.int32)
        model_apply_fn = TinyRecursiveMarkovianThinker(self.config, self.dtype).apply
        generated_bytes_tensor = generate_uncompiled_jax(ema_params, prompt_bytes, key, model_apply_fn, self.config, max_new_tokens=256)
        generated_bytes_tensor.block_until_ready()
        response_text = bytearray(np.array(generated_bytes_tensor)).decode('utf-8', errors='replace')
        display_text = Text()
        display_text.append("Prompt: ", style="bold yellow"); display_text.append(prompt_text + "\n\n")
        display_text.append("Response: ", style="bold cyan"); display_text.append(response_text, style="cyan")
        return Panel(display_text, title="🧠 Live Generation Preview (EMA)", border_style="magenta", expand=True)
    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB",f"{u.gpu}%"
        except Exception: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=60):
        if not data: return " " * w
        hist = np.array(list(data)); s=" ▂▃▄▅▆▇█"
        if hist.size < 2: return " " * w
        min_v, max_v = hist.min(), hist.max()
        if max_v == min_v or not np.isfinite(min_v) or not np.isfinite(max_v): return s[0] * len(hist) + " " * (w - len(hist))
        bins = np.linspace(min_v, max_v, len(s)); indices = np.clip(np.digitize(hist[-w:], bins) - 1, 0, len(s) - 1); return "".join(s[i] for i in indices)
    def _generate_layout(self) -> Layout:
        layout = Layout(); layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(self.progress, name="footer", size=3)); layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        precision = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"; header = f"🚀🧠 [bold]Grouped Ascended Thinker[/] | Groups: {self.config.NUM_GROUPS} | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"; layout["header"].update(Panel(Align.center(header), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))
        stats_tbl = Table.grid(expand=True, padding=(0, 1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right"); q_status = {0: "[blue]WARMUP", 1: "[green]IMPROVING", 2: "[yellow]STAGNATED", 3: "[red]REGRESSING", -1: "[dim]Disabled"}.get(self.shared_state.get('q_status_code', -1))
        stats_tbl.add_row("Steps/sec", f"[blue]{self.shared_state.get('steps_per_sec', 0.0):6.2f}[/] 🚀"); stats_tbl.add_row("LR (Q-Ctrl)", f"[green]{self.shared_state.get('learning_rate', 0.0):.2e}[/] {q_status}"); stats_tbl.add_row("GPU Mem/Util", f"[yellow]{self.shared_state.get('gpu_mem', 'N/A')}[/] / [yellow]{self.shared_state.get('gpu_util', 'N/A')}[/]")
        loss_tbl = Table.grid(expand=True, padding=(0,1)); loss_tbl.add_column(style="dim"); loss_tbl.add_column(justify="right", style="bright_white")
        if self.config.ENTROPIX_ENABLED:
            task_loss = self.shared_state.get('loss', 0.0) - self.shared_state.get('cognitive_loss', 0.0)
            status = "" if self.shared_state.get('cog_loss_active', False) else " ([dim]WARMUP[/])"; loss_tbl.add_row("Task Loss", f"{task_loss:7.4f}"); loss_tbl.add_row(f"Cognitive Loss{status}", f"{self.shared_state.get('cognitive_loss', 0.0):7.4f}")
        loss_tbl.add_row("[bold]Total Loss[/]", f"[bold]{self.shared_state.get('loss', 0.0):7.4f}[/]")
        entropix_tbl = Table.grid(expand=True, padding=(0,1)); entropix_tbl.add_column(); entropix_tbl.add_column(justify="right"); entropix_tbl.add_column(width=6)
        if self.config.ENTROPIX_ENABLED:
            entropix_tbl.add_row("Naked Entropy", f"{self.shared_state.get('naked_entropy', 0.0):7.4f}", "[cyan]H[/]"); entropix_tbl.add_row("Naked Varentropy", f"{self.shared_state.get('naked_varentropy', 0.0):7.4f}", "[magenta]V[/]"); entropix_tbl.add_row("Chaos Metric", f"{self.shared_state.get('chaos_metric', 0.0):7.4f}", "[red]C[/]")
        pid_tbl = Table.grid(padding=(0,1)); pid_tbl.add_column(style="dim"); pid_tbl.add_column(justify="right", style="green")
        if self.config.USE_PID_CONTROLLERS:
            pid_tbl.add_row("λ Chaos:", f"{self.shared_state.get('chaos_weight', 0.0):.4f}"); pid_tbl.add_row("λ Divergence:", f"{self.shared_state.get('divergence_weight', 0.0):.4f}")
        layout["left"].update(Align.center(Group(Panel(stats_tbl, title="[bold]📊 Core Stats[/]"), Panel(loss_tbl, title="[bold]⚖️ Loss Components[/]", border_style="yellow"), Panel(entropix_tbl, title="[bold]🧠 Cognitive Metrics[/]", border_style="cyan"), Panel(pid_tbl, title="[bold] PID Weights[/]", border_style="green"))))
        sparks = Group(*[Panel(Align.center(f"[cyan]{self._get_sparkline(hist)}[/]"), title=f"{name.title()} Loss History", height=3, border_style="dim") for name, hist in self.shared_state.get('loss_history', {}).items()])
        prompt_panel = Panel(Text(self.shared_state.get('preview_prompt',""), justify="center", overflow="fold"), title="[bold]Live Preview Prompt (g)[/]", border_style="green")
        right_content = self.shared_state.get('live_preview_panel') or Text("... press 'g' to generate a preview ...", justify="center")
        layout["right"].update(Group(sparks, prompt_panel, right_content)); return layout
    def _ui_update_loop(self, live: Live):
        while not self.interactive_state.shutdown_event.is_set():
            if self.active_preview_future and self.active_preview_future.done():
                try: self.shared_state['live_preview_panel'] = self.active_preview_future.result()
                except Exception as e: self.shared_state['live_preview_panel'] = Panel(f"Preview Error: {e}", border_style="red")
                self.active_preview_future = None
            mem, util = self._get_gpu_stats(); self.shared_state['gpu_mem'], self.shared_state['gpu_util'] = mem, util
            live.update(self._generate_layout(), refresh=True); time.sleep(1.0 / 15.0)
    def train(self):
        signal.signal(signal.SIGINT, self.shutdown); signal.signal(signal.SIGTERM, self.shutdown)
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        assert self.config.D_MODEL % self.config.NUM_GROUPS == 0, "D_MODEL must be divisible by NUM_GROUPS"
        assert self.config.NUM_HEADS % self.config.NUM_GROUPS == 0, "NUM_HEADS must be divisible by NUM_GROUPS"
        assert self.config.D_FF % self.config.NUM_GROUPS == 0, "D_FF must be divisible by NUM_GROUPS"
        raw_text_path = self._prepare_data()
        dirichlet_support = TinyRecursiveMarkovianThinker.get_dirichlet_support(self.config)
        model = TinyRecursiveMarkovianThinker(self.config, self.dtype)
        key = jax.random.PRNGKey(42); main_key, params_key, dropout_key = jax.random.split(key, 3)
        dummy_indices = jnp.zeros((self.config.BATCH_SIZE, self.config.CHUNK_SIZE), dtype=jnp.int32)
        dummy_y = jnp.zeros((self.config.BATCH_SIZE, self.config.CHUNK_SIZE, self.config.D_MODEL), dtype=self.dtype)
        temp_entropix_engine = EntropixEngine(self.config, dirichlet_support=dirichlet_support)
        dummy_ds_state = temp_entropix_engine.initialize_state(self.config.BATCH_SIZE, self.dtype)
        params = model.init({'params': params_key, 'dropout': dropout_key}, dummy_indices, dummy_y, dummy_y, dummy_ds_state, global_step=0)['params']
        self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        self.console.print(f'--- 🚀 [bold]Grouped Ascended Thinker[/bold] initialized with [yellow]{self.param_count:,}[/yellow] parameters. ---', style="magenta")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.config.LEARNING_RATE, b1=0.9, b2=0.95))
        q_state = init_q_controller(Q_CONTROLLER_CONFIG_TEXT) if self.config.USE_Q_CONTROLLER else None
        pid_chaos_state = PIDState(0.0, 0.0, self.config.INITIAL_COGNITIVE_LOSS_WEIGHT)
        pid_div_state = PIDState(0.0, 0.0, self.config.INITIAL_COGNITIVE_LOSS_WEIGHT)
        state = JaxTrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer,
            ema_params=params, q_state=q_state, pid_chaos_state=pid_chaos_state, 
            pid_divergence_state=pid_div_state, dropout_key=dropout_key
        )
        ckpt_path = Path(self.config.CHECKPOINT_DIR) / f"{self.config.BASENAME}.ckpt" 
        ckpt_path.parent.mkdir(exist_ok=True)
        if ckpt_path.exists() and not self.config.FRESH_START:
            self.console.print(f"--- Resuming from checkpoint: [cyan]{ckpt_path}[/cyan] ---", style="blue")
            serialized_state = ckpt_path.read_bytes()
            restored_state = serialization.from_bytes(state, serialized_state)
            state = restored_state
            self.console.print(f"--- Resumed from step {int(state.step)}. ---", style="blue")
        jitted_update_step = self.create_jitted_update_step(model, dirichlet_support)
        dataset, num_examples = create_direct_text_dataset(raw_text_path, self.config.BLOCK_SIZE, self.config.BATCH_SIZE, self.config.REBATCH_SIZE, is_training=True)
        data_iterator = iter(tfds.as_numpy(dataset))
        try:
            with open(raw_text_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview_text = f.read(50 * (256 + 1))
            self.preview_data_buffer = [preview_text[i:i+256] for i in range(0, len(preview_text) - 256, 257)]
            self.console.print(f"✅ Loaded {len(self.preview_data_buffer)} validation samples for preview.", style="green")
        except Exception as e:
            self.console.print(f"[bold red]Error loading preview data: {e}[/bold red]")
            self.preview_data_buffer = ["The quick brown fox jumps over the lazy dog."]
        steps_per_epoch = num_examples // self.config.BATCH_SIZE
        total_steps = self.config.EPOCHS * steps_per_epoch
        progress_text_column = TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}")
        self.progress = Progress(progress_text_column, "[progress.percentage]{task.percentage:>3.0f}%", BarColumn(), "•", TextColumn("Step {task.completed}/{task.total}"), "•", TimeRemainingColumn(), transient=True)
        epoch_task = self.progress.add_task("Training", total=total_steps, completed=int(state.step), epoch=(int(state.step) // steps_per_epoch)+1, total_epochs=self.config.EPOCHS)
        self.shared_state['loss_history'] = {'total': deque(maxlen=200), 'task': deque(maxlen=200), 'cognitive': deque(maxlen=200)}
        last_step_time = time.time()
        live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        ui_thread = threading.Thread(target=self._ui_update_loop, args=(live,), daemon=True)
        try:
            live.start(); ui_thread.start()
            self.console.print("--- Compiling JAX training step (one-time cost)... ---")
            (compile_x, compile_y) = next(data_iterator)
            compile_x = compile_x.reshape(self.config.REBATCH_SIZE, self.config.BATCH_SIZE, self.config.BLOCK_SIZE)
            compile_y = compile_y.reshape(self.config.REBATCH_SIZE, self.config.BATCH_SIZE, self.config.BLOCK_SIZE)
            jitted_update_step(state, compile_x[0], compile_y[0])
            state.params['y_init'].block_until_ready() 
            self.console.print("--- ✅ Compilation complete. Starting training loop. ---")
            while int(state.step) < total_steps and not self.interactive_state.shutdown_event.is_set():
                super_batch_x_np, super_batch_y_np = next(data_iterator)
                super_batch_x_np = super_batch_x_np.reshape(self.config.REBATCH_SIZE, self.config.BATCH_SIZE, self.config.BLOCK_SIZE)
                super_batch_y_np = super_batch_y_np.reshape(self.config.REBATCH_SIZE, self.config.BATCH_SIZE, self.config.BLOCK_SIZE)
                for i in range(self.config.REBATCH_SIZE):
                    if self.interactive_state.shutdown_event.is_set(): break
                    batch_x_np = super_batch_x_np[i]
                    batch_y_np = super_batch_y_np[i]
                    state, metrics = jitted_update_step(state, batch_x_np, batch_y_np)
                    time_now = time.time()
                    self.shared_state['steps_per_sec'] = 1.0 / (time_now - last_step_time + 1e-9)
                    last_step_time = time_now
                    metrics_np = jax.device_get(metrics)
                    self.shared_state.update({k: v.item() for k, v in metrics_np.items()})
                    if self.config.USE_Q_CONTROLLER: self.shared_state['q_status_code'] = int(state.q_state.status_code)
                    loss_item, cog_loss_item = metrics_np['loss'], metrics_np['cognitive_loss']
                    if np.isfinite(loss_item):
                        self.shared_state['loss_history']['total'].append(loss_item)
                        self.shared_state['loss_history']['task'].append(loss_item - cog_loss_item)
                        self.shared_state['loss_history']['cognitive'].append(cog_loss_item)
                    current_step = int(state.step)
                    self.progress.update(epoch_task, completed=current_step, epoch=(current_step // steps_per_epoch) + 1)
                    force_save = self.interactive_state.get_and_reset_force_save()
                    force_preview = self.interactive_state.get_and_reset_force_preview()
                    if force_save or (current_step > 0 and current_step % self.config.SAVE_EVERY == 0): self._save_checkpoint(state, ckpt_path)
                    if force_preview and (self.active_preview_future is None or self.active_preview_future.done()):
                        self.preview_index = (self.preview_index + 1) % len(self.preview_data_buffer)
                        prompt = self.preview_data_buffer[self.preview_index]
                        self.shared_state['preview_prompt'] = prompt
                        self.shared_state['live_preview_panel'] = Panel(Align.center("Generating..."), title="🧠 Live Generation Preview (EMA)", border_style="yellow")
                        preview_key, main_key = jax.random.split(main_key)
                        self.active_preview_future = self.async_pool.submit(self._run_live_generation_preview, jax.device_get(state.ema_params), prompt, preview_key)
        finally:
            self.shutdown(); live.stop()
            self.console.print("\n--- Training loop terminated. Saving final state... ---")
            if 'state' in locals(): self._save_checkpoint(state, ckpt_path)
            ui_thread.join(timeout=1); key_listener_thread.join(timeout=1)
    def create_jitted_update_step(self, model, dirichlet_support):
        @partial(jit)
        def update_step_fn(state: JaxTrainState, batch_x: chex.Array, batch_y: chex.Array):
            dropout_key, q_key, new_base_key = jax.random.split(state.dropout_key, 3)
            if self.config.USE_Q_CONTROLLER:
                q_state_pre_action = q_controller_choose_action(state.q_state, q_key, self.config.LEARNING_RATE)
                lr = q_state_pre_action.current_lr
            else:
                q_state_pre_action = state.q_state
                lr = jnp.array(self.config.LEARNING_RATE)
            def loss_fn(params):
                temp_engine = EntropixEngine(config=self.config, dirichlet_support=dirichlet_support)
                current_batch_size = batch_x.shape[0]
                y = jnp.broadcast_to(params['y_init'], (current_batch_size, self.config.CHUNK_SIZE, self.config.D_MODEL))
                z = jnp.broadcast_to(params['z_init'], (current_batch_size, self.config.CHUNK_SIZE, self.config.D_MODEL))
                ds_state = temp_engine.initialize_state(current_batch_size, self.dtype)
                num_chunks = self.config.BLOCK_SIZE // self.config.CHUNK_SIZE
                def chunk_scan_fn(carry, chunk_idx):
                    y_prev, z_prev, ds_state_prev, pid_chaos_prev, pid_div_prev = carry
                    y_in, z_in, ds_state_in = map(jax.lax.stop_gradient, (y_prev, z_prev, ds_state_prev))
                    chunk_start = chunk_idx * self.config.CHUNK_SIZE
                    indices = jax.lax.dynamic_slice_in_dim(batch_x, chunk_start, self.config.CHUNK_SIZE, axis=1)
                    targets = jax.lax.dynamic_slice_in_dim(batch_y, chunk_start, self.config.CHUNK_SIZE, axis=1)
                    (y_final, z_final), logits, cog_metrics, loss_comp, ds_state_new = model.apply(
                        {'params': params}, indices, y_in, z_in, ds_state_in, 
                        global_step=state.step, rngs={'dropout': dropout_key}
                    )
                    task_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, self.config.VOCAB_SIZE), targets.reshape(-1)))
                    def update_cognitive_loss(carry):
                        c_m, l_c, chaos_pid, div_pid = carry
                        new_pid_chaos, chaos_w = pid_update(chaos_pid, c_m.chaos_metric.mean(), self.pid_controllers['chaos'][0], self.pid_controllers['chaos'][1], self.config.PID_WEIGHT_LIMIT)
                        new_pid_div, div_w = pid_update(div_pid, c_m.kl_divergence.mean(), self.pid_controllers['divergence'][0], self.pid_controllers['divergence'][1], self.config.PID_WEIGHT_LIMIT)
                        return chaos_w * l_c.loss_chaos + div_w * l_c.loss_divergence, new_pid_chaos, new_pid_div, chaos_w, div_w, c_m
                    is_cog_active = self.config.ENTROPIX_ENABLED and self.config.USE_COGNITIVE_LOSS and (state.step > self.config.COGNITIVE_LOSS_WARMUP_STEPS)
                    cognitive_loss, pid_chaos_new, pid_div_new, chaos_w, div_w, cog_metrics_out = jax.lax.cond(
                        is_cog_active, update_cognitive_loss,
                        lambda carry: (jnp.zeros((), dtype=task_loss.dtype), carry[2], carry[3], jnp.zeros((), dtype=task_loss.dtype), jnp.zeros((), dtype=task_loss.dtype), carry[0]),
                        (cog_metrics, loss_comp, pid_chaos_prev, pid_div_prev)
                    )
                    return (y_final, z_final, ds_state_new, pid_chaos_new, pid_div_new), (task_loss + cognitive_loss, cognitive_loss, chaos_w, div_w, cog_metrics_out)
                initial_carry = (y, z, ds_state, state.pid_chaos_state, state.pid_divergence_state)
                ( _, _, _, final_chaos_pid, final_div_pid), collected_metrics = jax.lax.scan(
                    chunk_scan_fn, initial_carry, jnp.arange(num_chunks)
                )
                chunk_losses, chunk_cog_losses, final_chaos_w, final_div_w, last_cog_metrics = collected_metrics
                avg_loss, avg_cog_loss = jnp.mean(chunk_losses), jnp.mean(chunk_cog_losses)
                aux_from_loss = {'pid_chaos_state': final_chaos_pid, 'pid_divergence_state': final_div_pid}
                metrics_dict = {
                    'loss': avg_loss, 'cognitive_loss': avg_cog_loss,
                    'chaos_weight': final_chaos_w[-1], 'divergence_weight': final_div_w[-1],
                    'naked_entropy': jnp.mean(last_cog_metrics.naked_entropy[-1]),
                    'naked_varentropy': jnp.mean(last_cog_metrics.naked_varentropy[-1]),
                    'kl_divergence': jnp.mean(last_cog_metrics.kl_divergence[-1]),
                    'chaos_metric': jnp.mean(last_cog_metrics.chaos_metric[-1]),
                    'cog_loss_active': (state.step > self.config.COGNITIVE_LOSS_WARMUP_STEPS).astype(self.dtype)
                }
                return avg_loss, (aux_from_loss, metrics_dict)
            (loss, (aux_from_loss, metrics)), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=lr)
            new_params = optax.apply_updates(state.params, updates)
            new_ema_params = jax.tree_util.tree_map(
                lambda ema, p: ema * self.config.EMA_DECAY + p * (1.0 - self.config.EMA_DECAY),
                state.ema_params, new_params
            )
            if self.config.USE_Q_CONTROLLER:
                final_q_state = q_controller_update(q_state_pre_action, metrics['loss'])
            else:
                final_q_state = state.q_state
            new_state = state.replace(
                step=state.step + 1, params=new_params, ema_params=new_ema_params,
                opt_state=new_opt_state, q_state=final_q_state,
                pid_chaos_state=aux_from_loss['pid_chaos_state'],
                pid_divergence_state=aux_from_loss['pid_divergence_state'],
                dropout_key=new_base_key
            )
            metrics['learning_rate'] = lr
            return new_state, metrics
        return update_step_fn
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ascended Thinker v2.1 (JAX): Advanced Trainer")
    cfg = Config()
    parser.add_argument('--data-dir', type=str, default=cfg.DATA_DIR)
    parser.add_argument('--raw-text-file', type=str, default=cfg.RAW_TEXT_FILE)
    parser.add_argument('--basename', type=str, default=cfg.BASENAME)
    parser.add_argument('--block-size', type=int, default=cfg.BLOCK_SIZE)
    parser.add_argument('--chunk-size', type=int, default=cfg.CHUNK_SIZE)
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--rebatch-size', type=int, default=cfg.REBATCH_SIZE, help="Number of batches to load into memory at once to maximize GPU throughput.")
    parser.add_argument('--learning-rate', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--disable-entropix', action='store_true')
    parser.add_argument('--num-groups', type=int, default=cfg.NUM_GROUPS, help="Number of parallel thought groups.")
    args = parser.parse_args()
    for arg, value in vars(args).items():
        if value is not None: setattr(cfg, arg.upper().replace('-', '_'), value)
    if cfg.BLOCK_SIZE % cfg.CHUNK_SIZE != 0:
        print(f"[FATAL] --block-size ({cfg.BLOCK_SIZE}) must be divisible by --chunk-size ({cfg.CHUNK_SIZE}).")
        sys.exit(1)
    if args.disable_entropix: cfg.ENTROPIX_ENABLED = False
    trainer = SotaTrainerJAX(cfg)
    trainer.train()