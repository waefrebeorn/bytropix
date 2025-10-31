import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
except NameError:
    cache_dir = os.path.join(os.path.expanduser("~"), ".jax_cache_wubu_physics")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir

import sys
import time
from pathlib import Path
import signal
import threading
import platform
from collections import deque
import queue
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any, List, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import jax.scipy.special as jsp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax import struct, serialization
from flax.core import freeze, unfreeze
import chex

print("--- Verifying Dependencies ---")
try:
    import tensorflow
    import rich
    import pynvml
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
except ImportError:
    print("\n[FATAL] Missing one or more core dependencies. Please run: pip install tensorflow rich nvidia-ml-py tokenizers")
    sys.exit(1)
print("--- All dependencies verified. ---")

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.console import Group, Console
from rich.align import Align
from rich.text import Text
from rich.padding import Padding
pynvml.nvmlInit()

jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_threefry_partitionable', True)

# ==============================================================================
# SECTION I: THE NEW PHYSICS ENGINE
# ==============================================================================

class DecomposedGradient(struct.PyTreeNode):
    remainders: optax.Updates
    quotients: optax.Updates

def decompose_gradient_pytree(updates: optax.Updates) -> DecomposedGradient:
    boundary = 2 * jnp.pi
    remainders_pytree = jax.tree_util.tree_map(lambda g: jnp.mod(g + jnp.pi, boundary) - jnp.pi, updates)
    quotients_pytree = jax.tree_util.tree_map(lambda g: jnp.floor((g + jnp.pi) / boundary).astype(jnp.int32), updates)
    return DecomposedGradient(remainders=remainders_pytree, quotients=quotients_pytree)

class WubuOptimizerState(struct.PyTreeNode):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates

def wubu_optimizer(learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> WubuOptimizerState:
        return WubuOptimizerState(count=jnp.zeros([], jnp.int32), moment1=jax.tree_util.tree_map(jnp.zeros_like, params), moment2=jax.tree_util.tree_map(jnp.zeros_like, params))
    def update_fn(updates: optax.Updates, state: WubuOptimizerState, params: optax.Params | None = None) -> tuple[optax.Updates, WubuOptimizerState]:
        decomposed = decompose_gradient_pytree(updates)
        new_moment1 = optax.incremental_update(decomposed.remainders, state.moment1, beta1)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, beta2)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, beta1, count)
        m2_hat = optax.bias_correction(new_moment2, beta2, count)
        final_updates = jax.tree_util.tree_map(lambda m1, m2: learning_rate * m1 / (jnp.sqrt(m2) + epsilon), m1_hat, m2_hat)
        return final_updates, WubuOptimizerState(count=count, moment1=new_moment1, moment2=new_moment2)
    return optax.GradientTransformation(init_fn, update_fn)

@struct.dataclass
class ManifoldPatch:
    position: chex.Array; orientation: chex.Array; tension: chex.Array

@struct.dataclass
class WorldState:
    patches: ManifoldPatch; total_energy: chex.Array

def toroidal_normalize_patches(patch: ManifoldPatch, num_patches: int, d_model: int) -> ManifoldPatch:
    normalized_position = jnp.tanh(patch.position)
    raw_orientation = jnp.mod(patch.orientation + jnp.pi, 2 * jnp.pi) - jnp.pi
    ori_norms = jnp.linalg.norm(raw_orientation, axis=-1, keepdims=True)
    normalized_orientation = jnp.where(ori_norms > 1e-6, raw_orientation / ori_norms, jnp.zeros_like(raw_orientation))
    activated_tension = nn.gelu(patch.tension)
    target_norm = jnp.sqrt(float(num_patches * d_model))
    ten_norms = jnp.linalg.norm(activated_tension, axis=(1, 2), keepdims=True)
    normalized_tension = jnp.where(ten_norms > 1e-6, (activated_tension / ten_norms) * target_norm, jnp.zeros_like(activated_tension))
    return ManifoldPatch(position=normalized_position, orientation=normalized_orientation, tension=normalized_tension)

def safe_cosine_similarity(vec1, vec2_b, epsilon=1e-8):
    if vec2_b.ndim == 1: vec2 = vec2_b[None, None, :]
    else: vec2 = vec2_b[:, None, :]
    norm1 = jnp.linalg.norm(vec1, axis=-1); norm2 = jnp.linalg.norm(vec2, axis=-1)
    dot_product = jnp.sum(vec1 * vec2, axis=-1)
    similarity = jnp.where((norm1 > epsilon) & (norm2 > epsilon), dot_product / (norm1 * norm2), 0.0)
    return similarity

class PhysicsCell(nn.Module):
    num_patches: int; d_model: int; dtype: Any
    @nn.compact
    def __call__(self, world_state: WorldState, token_embedding: chex.Array):
        token_emb_2d = jnp.atleast_2d(token_embedding); token_emb_b = token_emb_2d[:, None, :]
        token_orientation_2d = nn.Dense(features=self.d_model, dtype=self.dtype, name="token_to_orientation")(token_emb_2d)
        pos_interaction = jnp.sum(world_state.patches.position * token_emb_b, axis=-1)
        ori_interaction = safe_cosine_similarity(world_state.patches.orientation, token_orientation_2d)
        interaction_weights = nn.softmax(pos_interaction + ori_interaction, axis=-1)
        force = interaction_weights[..., None] * token_emb_b
        pos_update_mlp = nn.Dense(features=self.d_model, dtype=self.dtype, name="position_dynamics")
        ori_update_mlp = nn.Dense(features=self.d_model, dtype=self.dtype, name="orientation_dynamics")
        tension_update_mlp = nn.Dense(features=self.d_model, dtype=self.dtype, name="tension_dynamics")
        delta_pos = pos_update_mlp(world_state.patches.position + force)
        delta_ori = ori_update_mlp(world_state.patches.orientation + force)
        delta_tension = tension_update_mlp(world_state.patches.tension + force)
        raw_new_patch = ManifoldPatch(position=world_state.patches.position + delta_pos, orientation=world_state.patches.orientation + delta_ori, tension=world_state.patches.tension + delta_tension)
        new_patches = toroidal_normalize_patches(raw_new_patch, self.num_patches, self.d_model)
        new_total_energy = jnp.sum(new_patches.tension, axis=(1, 2))
        new_world_state = WorldState(patches=new_patches, total_energy=new_total_energy)
        return new_world_state, new_world_state

class EnergyMinimizingLM(nn.Module):
    config: 'Config'; dtype: Any
    def setup(self):
        cfg = self.config
        self.embedding = nn.Embed(cfg.EFFECTIVE_VOCAB_SIZE, cfg.D_MODEL, dtype=self.dtype, name="embedding")
        self.physics_cell = PhysicsCell(num_patches=cfg.NUM_PATCHES, d_model=cfg.D_MODEL, dtype=self.dtype, name="PhysicsCell")
    def __call__(self, tokens: chex.Array, deterministic: bool = False):
        scan_fn = lambda mdl, carry, x: mdl.physics_cell(carry, x)
        scanner = nn.scan(scan_fn, variable_broadcast='params', split_rngs={'params': False}, in_axes=0)
        cfg = self.config; token_embeddings = self.embedding(tokens)
        patches_phase = jnp.linspace(0, 2 * jnp.pi, cfg.NUM_PATCHES, endpoint=False); dims_phase = jnp.linspace(0, 2 * jnp.pi, cfg.D_MODEL, endpoint=False)
        initial_phase = patches_phase[:, None] + dims_phase[None, :]; pos_pattern = jnp.sin(initial_phase)[None, ...] * 0.02; ori_pattern = jnp.cos(initial_phase)[None, ...] * 0.02
        initial_patches = ManifoldPatch(position=jnp.tile(pos_pattern, (cfg.BATCH_SIZE, 1, 1)), orientation=jnp.tile(ori_pattern, (cfg.BATCH_SIZE, 1, 1)), tension=jnp.ones((cfg.BATCH_SIZE, cfg.NUM_PATCHES, cfg.D_MODEL), dtype=self.dtype))
        initial_state = WorldState(patches=initial_patches, total_energy=jnp.sum(initial_patches.tension, axis=(-1, -2)))
        final_state, all_states = scanner(self, initial_state, jnp.transpose(token_embeddings, (1, 0, 2)))
        return all_states.total_energy, final_state
    def step(self, world_state: WorldState, token_embedding: chex.Array):
        new_state, _ = self.physics_cell(world_state, token_embedding); return new_state

# ==============================================================================
# SECTION II: CONFIG & DECOUPLED SCAFFOLDING
# ==============================================================================

class Config:
    DATA_DIR = "./data"; RAW_TEXT_FILE = "open_orca_formatted.txt"; TOKENIZER_FILE = "bpe_tokenizer.json"; VOCAB_SIZE = 16384; EFFECTIVE_VOCAB_SIZE = 16384
    BASENAME = "WubuPhysicsEngine_v1"; D_MODEL = 512; NUM_PATCHES = 64; CHECKPOINT_DIR = "./checkpoints"; EPOCHS = 50000; WUBU_LR = 1e-4; USE_BFLOAT16 = True
    SAVE_EVERY = 2000; FRESH_START = False; BATCH_SIZE = 4; SEQUENCE_LENGTH = 256; SUPER_BATCH_SIZE = 16; TEMPERATURE = 0.8; TOP_K = 50; PREVIEW_EVERY_N_STEPS = 100
    USE_CONWAY_REGULARIZER = True; CONWAY_EVERY = 1; VOID_TOKEN_ID = -1; VOID_CHAR = '░'; SPECIAL_TOKENS_MAP = {'[VOID]': VOID_CHAR}
    CLARIFYING_QUESTION_TOKEN_ID = 2564

class InteractivityState:
    def __init__(self, config: Config):
        self.lock = threading.Lock(); self.shutdown_event = threading.Event()
        self.data_queue = queue.Queue(maxsize=4)
        self.ui_messages = deque(maxlen=20)
        self.is_prompting = False; self.current_prompt_text = ""; self.prompt_submitted = False; self.force_save = False
        self.preview_enabled = True; self.generation_cpu_params: Optional[Dict] = None
        self.generation_params_lock = threading.Lock(); self.generation_request = deque(maxlen=1)
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def get_submitted_prompt(self):
        with self.lock:
            if self.prompt_submitted: self.prompt_submitted = False; prompt = self.current_prompt_text; self.current_prompt_text = ""; return prompt
            return None
    def set_shutdown(self): self.shutdown_event.set()
    def get_latest_cpu_params(self):
        with self.generation_params_lock: return self.generation_cpu_params
    def set_latest_cpu_params(self, params):
        with self.generation_params_lock: self.generation_cpu_params = params

class CustomTrainState(train_state.TrainState): pass
@dataclass
class AnimationFrame:
    canvas: np.ndarray
    title: str = ""
    step: int = 0
    total_steps: int = 0
    tokens: List[int] = field(default_factory=list)
    timestamp: float = 0.0
class AnimationState:
    def __init__(self): self.lock = threading.Lock(); self.current_frame: Optional[AnimationFrame] = None; self.is_running = False
    def set_frame(self, frame: AnimationFrame):
        with self.lock: self.current_frame = frame
    def get_frame(self) -> Optional[AnimationFrame]:
        with self.lock: return self.current_frame
    def start_animation(self):
        with self.lock: self.is_running = True
    def stop_animation(self):
        with self.lock: self.is_running = False; self.current_frame = None
    def is_active(self) -> bool:
        with self.lock: return self.is_running

def apply_game_of_life_rules(grid: jnp.ndarray) -> jnp.ndarray:
    padded_grid = jnp.pad(grid, 1, mode='wrap')
    neighbors_count = sum(jnp.roll(jnp.roll(padded_grid, i, axis=0), j, axis=1) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0)[1:-1, 1:-1]
    survivors = ((grid == 1) & ((neighbors_count == 2) | (neighbors_count == 3))).astype(jnp.int32)
    new_births = ((grid == 0) & (neighbors_count == 3)).astype(jnp.int32)
    return survivors + new_births

class DataLoader(threading.Thread):
    def __init__(self, shared_state: InteractivityState, config: Config, tokenizer, file_path: Path):
        super().__init__(daemon=True, name="DataLoader")
        self.shared_state = shared_state
        self.config = config
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.file_handle = None
        self.file_size = 0
        try:
            self.file_handle = open(self.file_path, 'rb')
            self.file_size = os.fstat(self.file_handle.fileno()).st_size
        except (IOError, FileNotFoundError) as e:
            print(f"[FATAL] DataLoader could not open data file: {e}")
            sys.exit(1)

    def _get_batch_from_file(self):
        batch_tokens = []
        for _ in range(self.config.BATCH_SIZE):
            max_start_byte = self.file_size - (self.config.SEQUENCE_LENGTH * 4)
            start_byte = np.random.randint(0, max_start_byte) if max_start_byte > 0 else 0
            self.file_handle.seek(start_byte)
            text_chunk = self.file_handle.read(self.config.SEQUENCE_LENGTH * 5).decode('utf-8', 'ignore')
            tokens = self.tokenizer.encode(text_chunk).ids
            padding_needed = self.config.SEQUENCE_LENGTH - len(tokens)
            if padding_needed > 0:
                tokens.extend([self.config.VOID_TOKEN_ID] * padding_needed)
            batch_tokens.append(tokens[:self.config.SEQUENCE_LENGTH])
        return jnp.array(batch_tokens, dtype=jnp.int32)

    def run(self):
        while not self.shared_state.shutdown_event.is_set():
            batch = self._get_batch_from_file()
            self.shared_state.data_queue.put(batch)
    def close(self):
        if self.file_handle:
            self.file_handle.close()

class AsyncParamSynchronizer(threading.Thread):
    def __init__(self, shared_state: InteractivityState, get_latest_state_fn, sync_interval_seconds=5.0):
        super().__init__(daemon=True, name="ParamSynchronizer"); self.shared_state = shared_state
        self.get_latest_state_fn = get_latest_state_fn; self.sync_interval = sync_interval_seconds
    def run(self):
        while not self.shared_state.shutdown_event.is_set():
            try:
                latest_state = self.get_latest_state_fn()
                if latest_state:
                    params_cpu = jax.device_get(latest_state.params)
                    params_f32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params_cpu)
                    self.shared_state.set_latest_cpu_params(params_f32)
            except Exception as e: self.shared_state.ui_messages.append(f"[bold red]ParamSync Error: {e}[/bold red]")
            for _ in range(int(self.sync_interval * 10)):
                if self.shared_state.shutdown_event.is_set(): break
                time.sleep(0.1)

# ==============================================================================
# SECTION III: TRAINING & INFERENCE FLOW
# ==============================================================================

def _create_physics_train_step(model: EnergyMinimizingLM, config: Config, grid_size: int):
    @partial(jit, static_argnames=('apply_conway',))
    def train_step_fn(state: CustomTrainState, batch_tokens: chex.Array, apply_conway: bool):
        def loss_fn(params):
            energy_trajectory, final_state = model.apply({'params': params}, batch_tokens, deterministic=False)
            final_energy = energy_trajectory[-1]
            energy_jumps = jnp.diff(energy_trajectory, axis=0)
            instability_penalty = jnp.mean(nn.relu(energy_jumps))
            conway_penalty = 0.0
            if apply_conway:
                tension_grid = final_state.patches.tension.mean(axis=-1).reshape(config.BATCH_SIZE, grid_size, grid_size)
                living_cells = (tension_grid > jnp.mean(tension_grid, axis=(1,2), keepdims=True)).astype(jnp.int32)
                next_gen_cells = jax.vmap(apply_game_of_life_rules)(living_cells)
                amount_of_change = jnp.mean(jnp.abs(next_gen_cells - living_cells))
                conway_penalty = 1.0 / (amount_of_change + 1e-6)
            total_loss = jnp.mean(final_energy) + 0.1 * instability_penalty + 0.001 * conway_penalty
            metrics = {'loss': jnp.mean(final_energy), 'instability_penalty': instability_penalty,
                       'conway_penalty': conway_penalty, 'total_loss': total_loss}
            return total_loss, metrics
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        (loss_val, metrics), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        metrics['grad_norm'] = optax.global_norm(grads)
        return new_state, metrics
    return train_step_fn

# ==============================================================================
# SECTION IV: ENTROPIX COGNITIVE SAMPLER
# ==============================================================================

# --- Entropix Config ---
EPS = 1e-8
MIN_TEMP = 0.01
MAX_TEMP = 100.0

# --- FIX: Use flax.struct.field to mark non-array attributes as static ---
@struct.dataclass
class Bilinear:
    # Dynamic array leaves
    bilinear: jnp.ndarray = field(default_factory=lambda: jnp.array(
        [
            [-0.033, -0.05, -0.1, -0.1],
            [-0.033, -0.05, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1],
        ]
    ))
    linear_state_ent: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0]))
    linear_state_std: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0]))
    # Static parameters
    linear_naked_ent: float = struct.field(pytree_node=False, default=-0.3)
    linear_naked_varent: float = struct.field(pytree_node=False, default=0.0)
    bias: float = struct.field(pytree_node=False, default=0.5)

@struct.dataclass
class Linear:
    # Static parameters
    weight: float = struct.field(pytree_node=False, default=0.0)
    bias: float = struct.field(pytree_node=False, default=0.3)

@struct.dataclass
class TargetEntropy:
    # Dynamic array leaves
    linear: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0]))
    linear_inv_temp: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0]))
    # Static parameters
    bias: float = struct.field(pytree_node=False, default=0.0)

@struct.dataclass
class DSConfig:
    # PyTree nodes (dynamic leaves)
    dirichlet_support: jnp.ndarray = field(default_factory=lambda: jnp.arange(128))
    outlier_threshold: Bilinear = field(default_factory=Bilinear)
    argmax_threshold: Linear = field(default_factory=Linear)
    dirichlet_threshold: Linear = field(default_factory=Linear)
    target_entropy: TargetEntropy = field(default_factory=TargetEntropy)

    # Static parameters (part of the treedef)
    outlier_topk: int = struct.field(pytree_node=False, default=128)
    emwa_ent_scaffold_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_ent_naked_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_varent_scaffold_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_varent_naked_coeff: float = struct.field(pytree_node=False, default=0.1)
    token_cross_ent_scaffold_coeff: float = struct.field(pytree_node=False, default=0.1)
    token_cross_ent_naked_coeff: float = struct.field(pytree_node=False, default=0.1)
    token_cross_var_scaffold_coeff: float = struct.field(pytree_node=False, default=0.1)
    token_cross_var_naked_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_topk_ent_naked_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_temp_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_dir_ent_coeff: float = struct.field(pytree_node=False, default=0.1)
    emwa_logp_base: float = struct.field(pytree_node=False, default=1.05)
    emwa_logp_exp_factor: float = struct.field(pytree_node=False, default=1.0)
    perturb_base_coeff: float = struct.field(pytree_node=False, default=1.1)
    perturb_exp_coeff: float = struct.field(pytree_node=False, default=1.0)
    noise_floor: float = struct.field(pytree_node=False, default=-18.0)
    
    
DEFAULT_DS_CONFIG = DSConfig()

# --- Entropix Sampler State & Functions ---
class DSState(NamedTuple):
  emwa_dir: jnp.ndarray; emwa_logp_on_supp: jnp.ndarray; emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray; emwa_ent_naked: jnp.ndarray; emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray; token_cross_ent_scaffold: jnp.ndarray; token_cross_ent_naked: jnp.ndarray
  token_cross_var_scaffold: jnp.ndarray; token_cross_var_naked: jnp.ndarray; emwa_dir_ent: jnp.ndarray
  emwa_topk_ent_naked: jnp.ndarray

@jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  p = jnp.exp(logp); ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]; varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent

@jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)

@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def fit_dirichlet(target_values, init_alpha=None, initial_lr=1.2, decay_alpha=0.1, decay_beta=2.0, decay_gamma=0.25, decay_nu=0.75, max_iters=140, tol=1e-4, dtype: jnp.dtype = jnp.bfloat16):
    batch_shape = target_values.shape[:-1]; n = target_values.shape[-1]; min_lr = 1e-8
    target_values = target_values.astype(jnp.float32)
    if init_alpha is None: init_alpha = jnp.ones((*batch_shape, n), dtype=jnp.float32)
    def halley_update(alpha, target_values):
        p1 = jsp.polygamma(1, alpha); p2 = jsp.polygamma(2, alpha); S = jnp.sum(alpha, axis=-1, keepdims=True)
        s1 = jsp.polygamma(1, S); s2 = jsp.polygamma(2, S); p1_inv = 1.0 / p1; sum_p1_inv = jnp.sum(p1_inv, axis=-1, keepdims=True)
        denom = 1.0 - s1 * sum_p1_inv; denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom); coeff = s1 / denom
        error = jsp.digamma(alpha) - jsp.digamma(S) - target_values; temp = p1_inv * error; sum_temp = jnp.sum(temp, axis=-1, keepdims=True)
        J_inv_error = temp + coeff * sum_temp * p1_inv; sum_J_inv_error = jnp.sum(J_inv_error, axis=-1, keepdims=True)
        H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error; temp2 = p1_inv * H_J_inv_error; sum_temp2 = jnp.sum(temp2, axis=-1, keepdims=True)
        J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
        return -J_inv_error + 0.5 * J_inv_H_J_inv_error
    def scan_body(carry, _):
        alpha, converged, error_norm, step = carry; S = jnp.sum(alpha, axis=-1, keepdims=True)
        error = jsp.digamma(alpha) - jsp.digamma(S) - target_values; error_norm = jnp.linalg.norm(error, axis=-1)
        new_converged = converged | (error_norm < tol); exp_factor = jnp.exp(-decay_alpha * (step**decay_nu))
        cos_factor = jnp.abs(jnp.cos(decay_beta / (step**decay_gamma))); lr = jnp.maximum(initial_lr * exp_factor * cos_factor, min_lr)
        delta_alpha = halley_update(alpha, target_values); scaled_delta_alpha = lr[..., None] * delta_alpha
        max_delta = 0.5 * alpha; scaled_delta_alpha = jnp.clip(scaled_delta_alpha, -max_delta, max_delta)
        new_alpha = jnp.where(new_converged[..., None], alpha, jnp.maximum(alpha + scaled_delta_alpha, alpha / 2))
        return (new_alpha, new_converged, error_norm, step + 1), None
    init_state = (init_alpha, jnp.zeros(batch_shape, dtype=jnp.bool_), jnp.full(batch_shape, jnp.inf), jnp.ones(batch_shape, dtype=jnp.int32))
    (final_alpha, final_converged, _, final_step), _ = jax.lax.scan(scan_body, init_state, None, length=max_iters)
    return final_alpha.astype(dtype), final_step - 1, final_converged

@jit
def dirichlet_log_likelihood_from_logprob(logprobs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    return (jnp.sum((alpha - 1.0) * logprobs, axis=-1) - jsp.gammaln(jnp.sum(alpha, axis=-1)) + jnp.sum(jsp.gammaln(alpha), axis=-1))

@jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old

@partial(jit, static_argnums=(3, 4, 5, 6))
def temp_tune(logits: jnp.ndarray, target_ent: jnp.ndarray, T_init: float = 1.0, lr: float = 0.1, max_iters: int = 10, tol: float = 1e-6, dtype: jnp.dtype = jnp.bfloat16):
    batch_size = logits.shape[0]; logits = logits.astype(jnp.float32)
    def ent_grad_hess(logits: jnp.ndarray, T: jnp.ndarray):
        p = jax.nn.softmax(logits / T[..., None], axis=-1); log_p = jax.nn.log_softmax(logits / T[..., None], axis=-1)
        mu1 = jnp.sum(p * log_p, axis=-1); diff = log_p - mu1[..., None]; mu2 = jnp.sum(p * diff**2, axis=-1)
        mu3 = jnp.sum(p * diff**3, axis=-1)
        return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)
    def scan_body(carry, _):
        T, iters, converged = carry; ent, grad, hess = ent_grad_hess(logits, T); error = ent - target_ent
        new_converged = converged | (jnp.abs(error) < tol); denominator = 2 * grad * grad - error * hess
        halley_step = jnp.where(jnp.abs(denominator) > 1e-8, 2 * error * grad / denominator, jnp.full_like(T, jnp.inf))
        newton_step = jnp.where(jnp.abs(grad) > 1e-8, error / grad, jnp.full_like(T, jnp.inf))
        grad_step = jnp.where(error > 0, lr * T, -lr * T)
        delta_T = jnp.where(jnp.abs(grad) < 1e-8, grad_step, jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step))
        delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
        new_T = jnp.where(new_converged, T, jnp.maximum(T - delta_T, T / 2))
        return (new_T, iters + 1, new_converged), None
    init_state = (jnp.full((batch_size,), T_init, dtype=jnp.float32), jnp.zeros(batch_size, dtype=jnp.int32), jnp.zeros(batch_size, dtype=jnp.bool_))
    (final_T, final_iters, final_converged), _ = jax.lax.scan(scan_body, init_state, None, length=max_iters)
    return final_T.astype(dtype), final_iters, final_converged

@partial(jit, static_argnames=("bsz", "dtype"))
def initialize_state(logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16) -> DSState:
  _, seqlen, _ = logits.shape; logprobs = normalize_logits(logits, config.noise_floor); ent, varent = ent_varent(logprobs)
  avg_ent, avg_varent = ent.mean(axis=-1), varent.mean(axis=-1); topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor); topk_ent, _ = ent_varent(topk_logprobs)
  avg_topk_ent = topk_ent.mean(axis=-1); logprobs_on_supp = normalize_logits(logits[..., config.dirichlet_support], config.noise_floor)
  avg_logprobs_on_supp = jnp.mean(logprobs_on_supp, axis=1); initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
  avg_dir_ent = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, initial_dir[:, None, :]).mean(axis=-1)
  topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1); initial_cross_ent_naked = -topk_token_logprobs.mean(axis=(1, 2))
  initial_cross_var_naked = topk_token_logprobs.var(axis=(1, 2))
  return DSState(emwa_dir=initial_dir.repeat(bsz, axis=0), emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, axis=0), emwa_temp=jnp.ones((bsz,), dtype=dtype), emwa_ent_scaffold=avg_ent.repeat(bsz, axis=0), emwa_ent_naked=avg_ent.repeat(bsz, axis=0), emwa_varent_scaffold=jnp.zeros((bsz,), dtype=dtype), emwa_varent_naked=avg_varent.repeat(bsz, axis=0), token_cross_ent_scaffold=avg_ent.repeat(bsz, axis=0), token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz, axis=0), token_cross_var_scaffold=jnp.zeros((bsz,), dtype=dtype), token_cross_var_naked=initial_cross_var_naked.repeat(bsz, axis=0), emwa_dir_ent=avg_dir_ent.repeat(bsz, axis=0), emwa_topk_ent_naked=avg_topk_ent.repeat(bsz, axis=0))

@partial(jit, static_argnames=("wild",))
def adaptive_dirichlet_step(key: jax.random.PRNGKey, state: DSState, logits: jnp.ndarray, config: DSConfig, wild: bool = True):
    dtype = logits.dtype; bsz, vsz = logits.shape; output_tokens = jnp.zeros(bsz, dtype=jnp.int32); EPS_ = jnp.array(1e-8, dtype=dtype)
    naked_log_probs = normalize_logits(logits, config.noise_floor); naked_ent, naked_varent = ent_varent(naked_log_probs)
    new_emwa_ent_naked = update_emwa(naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff)
    new_emwa_varent_naked = update_emwa(naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff)
    
    # Manually stack and add batch dim for einsum, as state leaves are scalar.
    state_ent = jnp.stack([state.token_cross_ent_scaffold, state.token_cross_ent_naked, state.emwa_ent_scaffold, state.emwa_ent_naked])[None, :]
    state_std = jnp.sqrt(jnp.stack([state.token_cross_var_scaffold, state.token_cross_var_naked, state.emwa_varent_scaffold, state.emwa_varent_naked]))[None, :]

    outlier_threshold = (jnp.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std) + jnp.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent) + jnp.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std) + naked_ent * config.outlier_threshold.linear_naked_ent + naked_varent * config.outlier_threshold.linear_naked_varent + config.outlier_threshold.bias)
    outlier_mask = outlier_threshold > 0; topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor); naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = update_emwa(naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff)
    argmax_threshold = (config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias)
    argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold); argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
    argmax_tokens = jnp.take_along_axis(topk_indices, argmax_indices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens); inlier_sampling_indices = ~outlier_mask & ~argmax_mask
    inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked); sampling_inlier_choices = jax.random.categorical(key, topk_logprobs / inlier_sampling_temp[:, None])
    sampling_inlier_tokens = jnp.take_along_axis(topk_indices, sampling_inlier_choices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(inlier_sampling_indices, sampling_inlier_tokens, output_tokens); target_entropy = (jnp.dot(state_ent.squeeze(0), config.target_entropy.linear) + jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1) + config.target_entropy.bias)
    temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy); new_emwa_temp = update_emwa(temp, state.emwa_temp, config.emwa_temp_coeff)
    tuned_logprobs = normalize_logits(naked_log_probs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP), config.noise_floor)
    logprobs_on_supp = normalize_logits(tuned_logprobs[:, config.dirichlet_support], config.noise_floor); kl = jnp.sum(jnp.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp), axis=-1)
    emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS_)); new_emwa_logp_on_supp = update_emwa(logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff[..., None])
    new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp); dir_log_likelihood = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, state.emwa_dir)
    new_emwa_dir_ent = update_emwa(-dir_log_likelihood, state.emwa_dir_ent, config.emwa_dir_ent_coeff); dirichlet_threshold = (config.dirichlet_threshold.weight * state.emwa_dir_ent + config.dirichlet_threshold.bias)
    use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
    gamma_samples = jax.random.gamma(key, new_emwa_dir, shape=new_emwa_dir.shape); dir_probs = gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)
    kl = jnp.sum(dir_probs * (jnp.log(dir_probs + EPS_) - logprobs_on_supp), axis=-1); perturb_coeff = 1 - jnp.pow(config.perturb_base_coeff, -config.perturb_exp_coeff * (1 / (kl + EPS_)))
    interpolated_probs = perturb_coeff[:, None] * dir_probs + (1 - perturb_coeff[:, None]) * jnp.exp(logprobs_on_supp)
    dicihlet_choices = jnp.argmax(interpolated_probs, axis=-1); dirichlet_tokens = jnp.take(config.dirichlet_support, dicihlet_choices)
    output_tokens = jnp.where(use_dirichlet, dirichlet_tokens, output_tokens); ood_choices = jax.random.categorical(key, jnp.log(dir_probs + EPS_))
    ood_tokens = jnp.take(config.dirichlet_support, ood_choices); output_tokens = jnp.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
    scaffold_ent, scaffold_varent = ent_varent(jnp.log(interpolated_probs + EPS_)); new_emwa_ent_scaffold = update_emwa(scaffold_ent, state.emwa_ent_scaffold, config.emwa_ent_scaffold_coeff)
    new_emwa_varent_scaffold = update_emwa(scaffold_varent, state.emwa_varent_scaffold, config.emwa_varent_scaffold_coeff)
    batch_indices = jnp.arange(bsz); scaffold_token_logprob = jnp.log(interpolated_probs[batch_indices, output_tokens] + EPS_)
    naked_token_logprob = jnp.log(jnp.take_along_axis(naked_log_probs, output_tokens[:, None], axis=-1).squeeze(axis=-1) + EPS_)
    token_cross_ent_naked = (config.token_cross_ent_naked_coeff * (-naked_token_logprob) + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked)
    token_cross_ent_scaffold = (config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob) + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold)
    token_cross_var_naked = (config.token_cross_var_naked_coeff * (token_cross_ent_naked - naked_token_logprob) ** 2 + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked)
    token_cross_var_scaffold = (config.token_cross_var_scaffold_coeff * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2 + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold)
    
    # --- FIX: Consistently squeeze all rank-1 results back to scalars for the output state ---
    # This ensures the state passed to a potential recursive call has a consistent shape.
    new_state = DSState(emwa_dir=new_emwa_dir.squeeze(axis=0),
                        emwa_logp_on_supp=new_emwa_logp_on_supp.squeeze(axis=0),
                        emwa_temp=new_emwa_temp.squeeze(axis=0),
                        emwa_ent_scaffold=new_emwa_ent_scaffold.squeeze(axis=0),
                        emwa_ent_naked=new_emwa_ent_naked.squeeze(axis=0),
                        emwa_varent_scaffold=new_emwa_varent_scaffold.squeeze(axis=0),
                        emwa_varent_naked=new_emwa_varent_naked.squeeze(axis=0),
                        token_cross_ent_scaffold=token_cross_ent_scaffold.squeeze(axis=0),
                        token_cross_ent_naked=token_cross_ent_naked.squeeze(axis=0),
                        token_cross_var_scaffold=token_cross_var_scaffold.squeeze(axis=0),
                        token_cross_var_naked=token_cross_var_naked.squeeze(axis=0),
                        emwa_dir_ent=new_emwa_dir_ent.squeeze(axis=0),
                        emwa_topk_ent_naked=new_emwa_topk_ent_naked.squeeze(axis=0))
                        
    return (new_state, output_tokens, naked_ent, naked_varent, scaffold_ent, scaffold_varent, naked_token_logprob, scaffold_token_logprob)






@dataclass(frozen=True)
class SamplerConfig:
    low_naked_entropy_threshold = 0.3; medium_naked_entropy_threshold = 1.2; high_naked_entropy_threshold = 2.5
    low_naked_varentropy_threshold = 1.2; high_naked_varentropy_threshold = 2.5
    low_scaffold_entropy_threshold = 1.0; high_scaffold_entropy_threshold = 2.0
    low_scaffold_varentropy_threshold = 0.3; high_scaffold_varentropy_threshold = 0.8

@partial(jit, static_argnames=("clarifying_question_token",))
def sample(state: DSState, logits: jnp.ndarray, config: DSConfig, clarifying_question_token: int, key=jax.random.PRNGKey(1337)):
    cfg = SamplerConfig()
    bsz = logits.shape[0]

    # This function is designed for inference where bsz is 1.
    # The original implementation used vmap, but for single-item generation,
    # a direct call is simpler and avoids shape issues with scalar states.
    if bsz != 1:
        # For simplicity in this context, we assert bsz is 1.
        # A batched implementation would require a lax.scan or fori_loop.
        raise ValueError(f"The generation loop's sample function expects a batch size of 1, but received {bsz}.")

    (new_state, new_token_arr, naked_ent, naked_varent, scaffold_ent, scaffold_varent, _, _) = adaptive_dirichlet_step(key, state, logits, config)

    # Squeeze the batch dimension from the results, since bsz=1
    single_logit = logits.squeeze(0)
    single_new_token = new_token_arr.squeeze(0)
    single_naked_ent = naked_ent.squeeze(0)
    single_naked_varent = naked_varent.squeeze(0)
    single_scaffold_ent = scaffold_ent.squeeze(0)
    single_scaffold_varent = scaffold_varent.squeeze(0)
    
    # This is the body of the original `sample_one` function
    def _and(*args): return jnp.bitwise_and.reduce(jnp.array(args))

    LELV = _and(single_naked_ent < cfg.low_naked_entropy_threshold, single_naked_varent < cfg.low_naked_varentropy_threshold).astype(float)
    HELV = _and(single_naked_ent > cfg.high_naked_entropy_threshold, single_naked_varent < cfg.low_naked_varentropy_threshold).astype(float)
    LEHV = _and(single_naked_ent < cfg.high_naked_entropy_threshold, single_naked_varent > cfg.high_naked_varentropy_threshold).astype(float)
    HEHV = _and(single_naked_ent > cfg.medium_naked_entropy_threshold, single_naked_varent > cfg.high_naked_varentropy_threshold).astype(float)
    
    case = jnp.argmax(jnp.hstack([LELV, HELV, LEHV, HEHV]))

    def lelv(): 
        return single_new_token, new_state
    
    def helv(): 
        return jnp.array(clarifying_question_token, dtype=single_new_token.dtype), new_state
        
    def lehv(): 
        return single_new_token, new_state
        
    def hehv():
        # Mask the token that was just chosen
        plogit = single_logit.at[single_new_token].set(float("-inf"))
        
        # The recursive call needs inputs with a batch dimension. `new_state` is scalar-leafed.
        (resampled_state, resampled_token_arr, *_) = adaptive_dirichlet_step(key, new_state, plogit[None, ...], DEFAULT_DS_CONFIG)
        
        # The result of the recursive call is batched, so we squeeze it back to a scalar.
        return resampled_token_arr.squeeze(0), resampled_state

    def default(): 
        return single_new_token, new_state

    branches = (lelv, helv, lehv, hehv, default)
    final_token, final_state = jax.lax.switch(case, branches)

    # Reshape the output to match the expected (batch, 1) shape.
    return final_token.reshape((1, 1)), final_state




# ==============================================================================
# SECTION V: THE MAIN TRAINER CLASS (INTEGRATED)
# ==============================================================================

class PhysicsTrainer:
    def __init__(self, config: Config):
        self.config = config; self.console = Console(); self.interactive_state: InteractivityState = None
        self.ui_lock = threading.Lock(); self.log_messages = deque(maxlen=15); self.last_metrics: Dict[str, Any] = {}
        self.steps_per_sec: float = 0.0; self.animation_state = AnimationState()
        self.dtype = jnp.bfloat16 if self.config.USE_BFLOAT16 and jax.devices('gpu') else jnp.float32
        self.param_count = 0; self.last_submitted_prompt = "A red apple on a wooden table."
        self.console.print("--- 🧩 Initializing BPE Tokenizer... ---", style="yellow")
        self.tokenizer, self.raw_text_path = self._prepare_data_and_tokenizer()

    def listen_for_keys(self):
        shared_state = self.interactive_state
        if platform.system() == "Windows": import msvcrt
        else: import sys, tty, termios, select
        try:
            if platform.system() == "Windows": tty_file = None; fd = -1; old_settings = None
            else: tty_file = open("/dev/tty"); fd = tty_file.fileno(); old_settings = termios.tcgetattr(fd); tty.setcbreak(fd)
            shared_state.ui_messages.append("--- Key listener active. Controls: [p] Prompt | [s] Save | [w] Preview | [q] Quit ---")
            while not shared_state.shutdown_event.is_set():
                key = None
                if platform.system() == "Windows":
                    if msvcrt.kbhit(): key = msvcrt.getch().decode('utf-8', 'ignore')
                else:
                    if select.select([tty_file], [], [], 0.05)[0]: key = tty_file.read(1)
                if key:
                    with shared_state.lock:
                        if shared_state.is_prompting:
                            if key in ['\r', '\n']: shared_state.is_prompting = False; shared_state.prompt_submitted = True
                            elif key == ('\x7f' if platform.system() != "Windows" else '\b'): shared_state.current_prompt_text = shared_state.current_prompt_text[:-1]
                            elif key.isprintable(): shared_state.current_prompt_text += key
                        else:
                            if key in ['q', '\x03']: shared_state.set_shutdown(); shared_state.ui_messages.append("Shutdown requested via keyboard."); break
                            elif key == 's': shared_state.force_save = True; shared_state.ui_messages.append("Manual save requested.")
                            elif key == 'p': shared_state.is_prompting = True
                            elif key == 'w':
                                shared_state.preview_enabled = not shared_state.preview_enabled
                                status = "[bold green]ENABLED[/]" if shared_state.preview_enabled else "[bold red]DISABLED[/]"
                                shared_state.ui_messages.append(f"☢️  Preview Window Toggled: {status}")
                else: time.sleep(0.05)
        except (IOError, FileNotFoundError): shared_state.ui_messages.append("[bold red]FATAL: Could not open TTY. Keyboard input is disabled.[/bold red]"); shared_state.shutdown_event.wait()
        finally:
            if platform.system() != "Windows" and 'old_settings' in locals() and old_settings is not None: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings); tty_file.close()

    def _log(self, message: str):
        with self.ui_lock: self.log_messages.append(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {message}")

    def _generator_thread_loop(self):
        while not self.interactive_state.shutdown_event.is_set():
            request = None
            if self.interactive_state.generation_request: request = self.interactive_state.generation_request.popleft()
            if request:
                params_f32, vocab_embeddings, prompt, key = request
                try: self._run_animated_preview(params_f32, vocab_embeddings, prompt, key)
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    self._log(f"[red]Animation thread error: {type(e).__name__}: {e}\n{tb_str}[/red]")
            time.sleep(0.5)

    def shutdown(self, signum=None, frame=None):
        if self.interactive_state and not self.interactive_state.shutdown_event.is_set():
            self._log("\n--- Shutdown signal received. Cleaning up... ---")
            self.interactive_state.set_shutdown()

    def _prepare_data_and_tokenizer(self):
        data_dir = Path(self.config.DATA_DIR); raw_path = data_dir / self.config.RAW_TEXT_FILE
        tokenizer_path = data_dir / self.config.TOKENIZER_FILE; data_dir.mkdir(exist_ok=True)
        if not raw_path.exists():
            self.console.print(f"❌ [bold red]FATAL: Data file not found at {raw_path}.[/bold red]"); sys.exit(1)
        special_tokens_list = list(self.config.SPECIAL_TOKENS_MAP.keys())
        if tokenizer_path.exists():
            self.console.print(f"✅ Found existing tokenizer at [cyan]{tokenizer_path}[/cyan]. Loading...")
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            self.console.print(f"⚠️  Tokenizer not found. Preparing to train a new one on [cyan]{raw_path}[/cyan]...")
            self.console.print("[dim]This may take several minutes for a large file. Please be patient.[/dim]")
            tokenizer = Tokenizer(models.BPE()); tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel(); trainer = trainers.BpeTrainer(vocab_size=self.config.VOCAB_SIZE, special_tokens=special_tokens_list)
            with self.console.status("[bold green]Training tokenizer...", spinner="earth"):
                tokenizer.train([str(raw_path)], trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            self.console.print(f"✅ Tokenizer training complete. Saved to [cyan]{tokenizer_path}[/cyan].")
        self.config.EFFECTIVE_VOCAB_SIZE = tokenizer.get_vocab_size(); self.config.VOCAB_SIZE = tokenizer.get_vocab_size()
        void_id = tokenizer.token_to_id('[VOID]');
        if void_id is not None: self.config.VOID_TOKEN_ID = void_id
        else: self.console.print("[bold red]FATAL: '[VOID]' token not found in tokenizer vocab.[/bold red]"); sys.exit(1)
        return tokenizer, raw_path

    @staticmethod
    @partial(jit, static_argnames=('apply_fn',))
    def _get_energy_logits(params: Dict, world_state: WorldState, vocab_embeddings: chex.Array, apply_fn) -> Tuple[chex.Array, WorldState]:
        """JIT-compiled function to perform the expensive vmap over the vocabulary."""
        def single_step_fn(ws, emb):
            return apply_fn({'params': params}, ws, emb, method=EnergyMinimizingLM.step)
        potential_next_states = jax.vmap(single_step_fn, in_axes=(None, 0))(world_state, vocab_embeddings)
        energy_logits = -potential_next_states.total_energy.squeeze()
        return energy_logits, potential_next_states

    def _run_animated_preview(self, params_f32, vocab_embeddings, prompt_text: str, key: chex.PRNGKey):
        cfg = self.config; model = EnergyMinimizingLM(self.config, dtype=jnp.float32)
        prompt_tokens = self.tokenizer.encode(prompt_text).ids
        initial_patches = ManifoldPatch(position=jnp.zeros((1, cfg.NUM_PATCHES, cfg.D_MODEL)), orientation=jnp.zeros((1, cfg.NUM_PATCHES, cfg.D_MODEL)), tension=jnp.ones((1, cfg.NUM_PATCHES, cfg.D_MODEL)))
        initial_total_energy = jnp.sum(initial_patches.tension, axis=(1, 2)); current_world_state = WorldState(patches=initial_patches, total_energy=initial_total_energy)
        prompt_embeddings = vocab_embeddings[jnp.array(prompt_tokens)]
        @jit
        def process_prompt(initial_state, embeddings):
            def scan_fn(state, token_emb): return model.apply({'params': params_f32}, state, token_emb, method=model.step), None
            final_state, _ = jax.lax.scan(scan_fn, initial_state, embeddings); return final_state
        current_world_state = process_prompt(current_world_state, prompt_embeddings)
        
        initial_logits, _ = self._get_energy_logits(params_f32, current_world_state, vocab_embeddings, model.apply)
        sampler_state = initialize_state(initial_logits[None, None, :], bsz=1, config=DEFAULT_DS_CONFIG, dtype=jnp.float32)
        sampler_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), sampler_state)

        generated_tokens = prompt_tokens[:]; self.animation_state.start_animation()
        try:
            for i in range(cfg.SEQUENCE_LENGTH - len(prompt_tokens)):
                # --- FIX: Added check for self.interactive_state.preview_enabled ---
                if not self.animation_state.is_active() or self.interactive_state.shutdown_event.is_set() or not self.interactive_state.preview_enabled:
                    break
                gen_key, key = jax.random.split(key)
                
                logits, potential_next_states = self._get_energy_logits(params_f32, current_world_state, vocab_embeddings, model.apply)
                
                next_token_array, sampler_state = sample(sampler_state, logits[None, :], config=DEFAULT_DS_CONFIG, clarifying_question_token=cfg.CLARIFYING_QUESTION_TOKEN_ID, key=gen_key)
                next_token_idx = next_token_array.flatten()[0]
                
                next_world_state = jax.tree_util.tree_map(lambda x: x[next_token_idx], potential_next_states)
                current_world_state = next_world_state
                
                generated_tokens.append(int(np.array(next_token_idx)))
                display_text = self.tokenizer.decode(generated_tokens)
                frame = AnimationFrame(
                    canvas=np.array([[display_text]]),
                    title=f" Physics-Based Generation ",
                    step=i + 1,
                    total_steps=cfg.SEQUENCE_LENGTH - len(prompt_tokens),
                    tokens=list(generated_tokens),
                    timestamp=time.time()
                )
                self.animation_state.set_frame(frame); time.sleep(1 / 24)
        finally: self.animation_state.stop_animation()

    def _save_checkpoint(self, state, path):
        path.parent.mkdir(exist_ok=True); state_cpu = jax.device_get(state); state_cpu = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16 else x, state_cpu); path.write_bytes(serialization.to_bytes(state_cpu)); self._log(f"💾 Checkpoint saved to [cyan]{path}[/cyan] (step {int(state.step)})")

    def _generate_layout(self, progress: Progress, global_step: int) -> Layout:
        with self.ui_lock:
            layout = Layout(); layout.split(Layout(name="header", size=3), Layout(ratio=3, name="main_content"), Layout(ratio=1, name="log_panel", minimum_size=5), Layout(name="footer", size=3)); layout["main_content"].split_row(Layout(name="left", minimum_size=40, ratio=1), Layout(name="right", ratio=2))
            with self.interactive_state.lock: preview_status = "[on bold green]ON[/]" if self.interactive_state.preview_enabled else "[on dim red]OFF[/]"
            precision = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"; header_text = f"🚀🧠 [bold]Wubu Physics Engine v1[/] | Step: {global_step} | SPS: {self.steps_per_sec:.2f} | Preview (w): {preview_status}"; layout["header"].update(Panel(Align.center(header_text), style="bold magenta", title=f"[dim]Params: {self.param_count/1e6:.2f}M | Precision: {precision}[/dim]", title_align="right"))
            mem, util = self._get_gpu_stats(); stats_tbl = Table.grid(expand=True, padding=(0, 1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right"); stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:6.2f}[/] 🚀"); stats_tbl.add_row("Wubu LR", f"[green]{self.config.WUBU_LR:.2e}[/]"); stats_tbl.add_row("GPU Mem/Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            loss_tbl = Table.grid(expand=True, padding=(0, 1)); loss_tbl.add_column(style="dim"); loss_tbl.add_column(justify="right", style="bright_white"); loss_tbl.add_row("Final Energy", f"{float(self.last_metrics.get('loss', 0)):7.4f}"); loss_tbl.add_row("Instability", f"{float(self.last_metrics.get('instability_penalty', 0)):7.4f}");
            if self.config.USE_CONWAY_REGULARIZER: loss_tbl.add_row("Conway Penalty", f"{float(self.last_metrics.get('conway_penalty', 0)):7.4f}")
            loss_tbl.add_row("[bold]Total Loss[/]", f"[bold]{float(self.last_metrics.get('total_loss', 0)):7.4f}[/]"); layout["left"].update(Align.center(Group(Panel(stats_tbl, title="[bold]📊 Core Stats[/]"), Panel(loss_tbl, title="[bold]⚡ Physics & Loss[/]", border_style="yellow"))))
            anim_frame = self.animation_state.get_frame()
            if anim_frame: display_text = anim_frame.canvas[0, 0]; panel = Panel(Text(display_text), title=anim_frame.title, border_style="magenta", height=20); anim_progress = Progress(TextColumn("[cyan]Generating[/]", justify="right"), BarColumn(bar_width=None), TextColumn("{task.completed}/{task.total}")); anim_progress.add_task("steps", total=anim_frame.total_steps, completed=anim_frame.step); right_content = Group(panel, Padding(anim_progress, (1, 0)))
            else: right_content = Panel(Align.center(Text("... waiting for preview generation ...", justify="center"), vertical="middle"), border_style="dim")
            layout["right"].update(right_content); log_text = Text("\n".join(self.log_messages), no_wrap=True); layout["log_panel"].update(Panel(log_text, title="[dim]Console Log[/dim]", border_style="dim"))
            with self.interactive_state.lock:
                if self.interactive_state.is_prompting: layout["footer"].update(Panel(Text(self.interactive_state.current_prompt_text + "█", justify="left"), title="[bold yellow]Enter prompt (Enter to submit)[/]", border_style="yellow"))
                elif progress: layout["footer"].update(Padding(progress, (1, 0)))
            return layout

    def _get_gpu_stats(self):
        try: handle = pynvml.nvmlDeviceGetHandleByIndex(0); mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle); util_info = pynvml.nvmlDeviceGetUtilizationRates(handle); return f"{mem_info.used / 1024**3:.2f}/{mem_info.total / 1024**3:.2f} GiB", f"{util_info.gpu}%"
        except Exception: return "N/A", "N/A"

    def handle_requests(self, state, ckpt_path, global_step):
        submitted_prompt = self.interactive_state.get_submitted_prompt()
        if submitted_prompt: self.last_submitted_prompt = submitted_prompt
        is_preview_step = (global_step > 0 and global_step % self.config.PREVIEW_EVERY_N_STEPS == 0)
        if self.interactive_state.preview_enabled and (submitted_prompt or is_preview_step):
            cpu_params = self.interactive_state.get_latest_cpu_params()
            if cpu_params and not self.interactive_state.generation_request:
                self._log("☢️  Queueing generation with latest async parameters...")
                vocab_embeddings = cpu_params['embedding']['embedding']
                preview_key = jax.random.PRNGKey(global_step)
                self.interactive_state.generation_request.append((cpu_params, vocab_embeddings, self.last_submitted_prompt, preview_key))
        if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and global_step % self.config.SAVE_EVERY == 0): self._save_checkpoint(state, ckpt_path)

    def train(self):
        signal.signal(signal.SIGINT, self.shutdown); signal.signal(signal.SIGTERM, self.shutdown)
        self.console.print("--- 🚀 [bold]Starting Physics Engine Training[/bold] ---", style="magenta")
        model = EnergyMinimizingLM(self.config, self.dtype); key = jax.random.PRNGKey(42)
        params = model.init({'params': jax.random.split(key)[0]}, jnp.zeros((1, self.config.SEQUENCE_LENGTH), dtype=jnp.int32))['params']
        self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(params)); self.console.print(f'--- [bold]Model[/bold] initialized with [yellow]{self.param_count:,}[/yellow] parameters. ---')
        optimizer = wubu_optimizer(self.config.WUBU_LR); state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        ckpt_path = Path(self.config.CHECKPOINT_DIR) / f"{self.config.BASENAME}.ckpt"
        if not self.config.FRESH_START and ckpt_path.exists():
            self.console.print(f"--- Resuming from checkpoint: [cyan]{ckpt_path}[/cyan] ---")
            with ckpt_path.open('rb') as f: state = serialization.from_bytes(state, f.read())
            self.console.print(f"--- Resumed from step {int(state.step)}. ---")
        try: file_size = os.path.getsize(self.raw_text_path)
        except FileNotFoundError: self.console.print(f"[bold red]FATAL: Data file not found at {self.raw_text_path} for size calculation.[/bold red]"); sys.exit(1)
        approx_sequences_in_dataset = file_size // (self.config.SEQUENCE_LENGTH * 2)
        steps_per_epoch = max(1, approx_sequences_in_dataset // self.config.BATCH_SIZE); total_steps = self.config.EPOCHS * steps_per_epoch
        self.console.print(f"--- Dataset Info: Approx. [cyan]{steps_per_epoch:,}[/cyan] steps per epoch. Training for [cyan]{self.config.EPOCHS}[/cyan] epochs ([cyan]{total_steps:,}[/cyan] total steps). ---")
        grid_size = int(np.sqrt(self.config.NUM_PATCHES)); jitted_train_step = _create_physics_train_step(model, self.config, grid_size)
        self.interactive_state = InteractivityState(self.config)
        data_loader = DataLoader(self.interactive_state, self.config, self.tokenizer, self.raw_text_path)
        self.console.print("--- Compiling JAX functions (one-time cost)... ---", style="yellow")
        first_batch = data_loader._get_batch_from_file()
        state, _ = jitted_train_step(state, first_batch, apply_conway=False)
        self.console.print("--- ✅ JAX compilation complete. ---", style="green")
        data_loader.start()
        latest_state_ref = [state]
        param_synchronizer = AsyncParamSynchronizer(self.interactive_state, lambda: latest_state_ref[0]); param_synchronizer.start()
        key_listener_thread = threading.Thread(target=self.listen_for_keys, daemon=True)
        generator_thread = threading.Thread(target=self._generator_thread_loop, daemon=True)
        global_step = int(state.step)
        progress = Progress(TextColumn("{task.description}"), BarColumn(), "•", TextColumn("Step {task.completed}/{task.total}"), "•", TimeRemainingColumn())
        main_task = progress.add_task(f"[bold]Epoch 1/{self.config.EPOCHS}[/]", total=total_steps, completed=global_step)
        live = Live(self._generate_layout(progress, global_step), screen=True, vertical_overflow="crop", auto_refresh=False)
        try:
            live.start()
            key_listener_thread.start(); generator_thread.start()
            self._log("--- ✅ All systems go. Starting training loop. ---")
            while not self.interactive_state.shutdown_event.is_set() and global_step < total_steps:
                step_start_time = time.time()
                batch = self.interactive_state.data_queue.get()
                apply_conway = self.config.USE_CONWAY_REGULARIZER and (global_step % self.config.CONWAY_EVERY == 0)
                state, metrics = jitted_train_step(state, batch, apply_conway=apply_conway)
                latest_state_ref[0] = state
                self.handle_requests(state, ckpt_path, global_step)
                self.steps_per_sec = 1.0 / (time.time() - step_start_time + 1e-9); self.last_metrics = jax.device_get(metrics)
                with self.interactive_state.lock:
                    while self.interactive_state.ui_messages: self._log(self.interactive_state.ui_messages.popleft())
                current_epoch = global_step // steps_per_epoch + 1
                progress.update(main_task, completed=global_step, description=f"[bold]Epoch {current_epoch}/{self.config.EPOCHS}[/]")
                live.update(self._generate_layout(progress, global_step), refresh=True)
                global_step += 1
        finally:
            self.shutdown(); live.stop()
            self.console.print("\n--- Training loop terminated. Saving final state... ---")
            if 'state' in locals(): self._save_checkpoint(state, ckpt_path)
            self.console.print("--- Final log messages ---")
            for msg in list(self.log_messages): self.console.print(Text.from_markup(msg))
            data_loader.close()
            key_listener_thread.join(timeout=1); generator_thread.join(timeout=1)






def main():
    parser = argparse.ArgumentParser(description="Wubu Physics Engine v1")
    cfg = Config()
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--data-dir', type=str, default=cfg.DATA_DIR)
    parser.add_argument('--fresh-start', action='store_true')
    args = parser.parse_args()
    for arg, value in vars(args).items():
        if hasattr(cfg, arg.upper()) and value is not None: setattr(cfg, arg.upper(), value)
    PhysicsTrainer(cfg).train()

if __name__ == "__main__":
    main()