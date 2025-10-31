import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
try:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
except NameError:
    cache_dir = os.path.join(os.path.expanduser("~"), ".jax_cache_wubu_genesis")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = cache_dir
import sys, time, signal, threading, platform, argparse
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Optional, Any, List
from functools import partial
from dataclasses import dataclass
import jax, jax.numpy as jnp, numpy as np, optax
from jax import jit, value_and_grad, lax
from flax import linen as nn, struct, serialization
from flax.training import train_state
import chex
import queue 
print("--- Verifying Dependencies ---");
try:
    import tensorflow, rich, pynvml
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    print("--- All dependencies verified. ---")
except ImportError as e:
    print(f"\n[FATAL] Missing a core dependency: {e}. Please run: pip install tensorflow rich nvidia-ml-py tokenizers")
    sys.exit(1)

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
# Add these imports at the top
import logging
from rich.logging import RichHandler
jax.config.update("jax_debug_nans", False) # Keep on for one more run to be sure
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_threefry_partitionable', True)

class Config:
    DATA_DIR = "./data"; RAW_TEXT_FILE = "open_orca_formatted.txt"; BASENAME = "WubuGenesis_v1"; CHECKPOINT_DIR = "./checkpoints"
    NUM_PAGES = 4
    CANVAS_WIDTH = 128; CANVAS_HEIGHT = 64
    D_MODEL = 256; PATCHES_PER_PAGE = 64
    CNN_FEATURES = [64, 128, 256, D_MODEL]; CNN_KERNEL_SIZE = (3, 3)
    EPOCHS = 1000; WUBU_LR = 1e-4; USE_BFLOAT16 = True 
    BATCH_SIZE = 1; SUPER_BATCH_SIZE = 1
    SIMULATION_STEPS = 8 
    USE_GRAMMAR_FIELD = True; NUM_GRAMMAR_EXPERTS = 4
    USE_COGNITIVE_OBSERVER = True; USE_NOVELTY_ENGINE = True
    USE_ODOMETER_STATE = True
    DISSIPATION_FACTOR = 0.95 
    FREE_ENERGY_TEMP = 0.1
    NOVELTY_HISTORY_LENGTH = 8
    PREVIEW_EVERY_N_STEPS = 1
    FRESH_START = False
    SAVE_EVERY = 5000

@struct.dataclass
class OdometerState:
    remainders: chex.Array
    quotients: chex.Array

@jit
def update_odometer_state(current_state: OdometerState, update_val: chex.Array) -> OdometerState:
    b = 2 * jnp.pi
    unwrapped_val = current_state.remainders + update_val
    new_remainders = jnp.mod(unwrapped_val + jnp.pi, b) - jnp.pi
    quotient_update = jnp.floor((unwrapped_val + jnp.pi) / b).astype(jnp.int32)
    new_quotients = current_state.quotients + quotient_update
    return OdometerState(remainders=new_remainders, quotients=new_quotients)

@jit
def angular_distance(y_pred: chex.Array, y_true: chex.Array) -> chex.Array:
    diff = y_pred - y_true
    return jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi

class PIDLambdaController:
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets; self.base_weights = base_weights; self.gains = gains
        self.state = {'integral_error': {k: 0.0 for k in targets}, 'last_error': {k: 0.0 for k in targets}}
    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        lambdas = {}
        for name, base_weight in self.base_weights.items():
            metric_name = f"loss_{name}"
            current_loss = last_metrics.get(metric_name)
            if current_loss is not None and metric_name in self.targets and np.isfinite(current_loss):
                kp,ki,kd=self.gains[metric_name]; target=self.targets[metric_name]; error=float(current_loss)-target
                p_term=kp*error; self.state['integral_error'][metric_name]=np.clip(self.state['integral_error'][metric_name]+error,-5.0,5.0)
                i_term=ki*self.state['integral_error'][metric_name]; derivative=error-self.state['last_error'][metric_name]; d_term=kd*derivative
                self.state['last_error'][metric_name]=error; lambdas[name]=float(np.clip(base_weight*np.exp(p_term+i_term+d_term),0.0,10.0))
            else: lambdas[name]=float(base_weight)
        return lambdas

@jax.jit
def _toroidally_limit_pytree(updates: optax.Updates, max_norm: float) -> optax.Updates:
    g_norm = optax.global_norm(updates)
    trigger = g_norm > max_norm
    wrapped_norm = jnp.fmod(g_norm, max_norm)
    scaling_factor = wrapped_norm / (g_norm + 1e-9)
    return jax.tree_util.tree_map(
        lambda g: jnp.where(trigger, g * scaling_factor, g), updates
    )

@struct.dataclass
class DecomposedGradient: remainders: optax.Updates; quotients: optax.Updates
def decompose_gradient_pytree(updates: optax.Updates) -> DecomposedGradient:
    b=2*jnp.pi
    remainders = jax.tree_util.tree_map(lambda g: jnp.mod(g + jnp.pi, b) - jnp.pi, updates)
    quotients = jax.tree_util.tree_map(lambda g: lax.stop_gradient(jnp.floor((g + jnp.pi) / b).astype(jnp.int32)), updates)
    return DecomposedGradient(remainders, quotients)

@struct.dataclass
class WubuOptimizerState: count: chex.Array; moment1: optax.Updates; moment2: optax.Updates
def wubu_optimizer(lr: float, b1=0.9, b2=0.999, eps=1e-8) -> optax.GradientTransformation:
    max_norm = jnp.pi 
    def init_fn(p: optax.Params)->WubuOptimizerState:
        return WubuOptimizerState(count=jnp.zeros([], jnp.int32), moment1=jax.tree_util.tree_map(jnp.zeros_like, p), moment2=jax.tree_util.tree_map(jnp.zeros_like, p))
    def update_fn(u: optax.Updates, s: WubuOptimizerState, p: optax.Params|None=None)->tuple[optax.Updates, WubuOptimizerState]:
        u_toroidally_limited = _toroidally_limit_pytree(u, max_norm)
        dec = decompose_gradient_pytree(u_toroidally_limited)
        m1 = optax.incremental_update(dec.remainders, s.moment1, b1)
        m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, u_toroidally_limited), s.moment2, b2)
        c = s.count+1
        m1_hat = optax.bias_correction(m1, b1, c)
        m2_hat = optax.bias_correction(m2, b2, c)
        final_u = jax.tree_util.tree_map(lambda m,v: lr * m / (jnp.sqrt(v) + eps), m1_hat, m2_hat)
        return final_u, WubuOptimizerState(c, m1, m2)
    return optax.GradientTransformation(init_fn, update_fn)

@dataclass
class CharacterEvent: char_code:int; num_bytes:int; is_letter:bool; is_number:bool; is_punctuation:bool; is_whitespace:bool
def deterministic_byte_parser_cpu(byte_chunk: np.ndarray, canvas_width: int, canvas_height: int) -> Dict[str, np.ndarray]:
    canvas_size = canvas_height * canvas_width
    if len(byte_chunk) < canvas_size: byte_chunk=np.pad(byte_chunk,(0,canvas_size-len(byte_chunk)))
    byte_chunk=byte_chunk[:canvas_size]; events=[]
    for byte in byte_chunk:
        is_l=(byte>=65 and byte<=90)or(byte>=97 and byte<=122); is_n=byte>=48 and byte<=57
        is_w=byte==32 or byte==10 or byte==9; is_p=not(is_l or is_n or is_w)
        events.append(CharacterEvent(byte,1,is_l,is_n,is_p,is_w))
    event_dict={f.name:np.array([getattr(e,f.name)for e in events], dtype=np.int64) for f in CharacterEvent.__dataclass_fields__.values()}
    for key in event_dict: event_dict[key]=event_dict[key].reshape(canvas_height,canvas_width)
    return event_dict

# --- DATA & MODEL STRUCTURES (safe_cosine_similarity is corrected) ---

@struct.dataclass
class ManifoldPatch:
    position: OdometerState
    orientation: chex.Array
    tension: OdometerState

@struct.dataclass
class WorldState:
    pages: ManifoldPatch
    total_energy: chex.Array

@struct.dataclass
class PhysicsStepOutput:
    pages: ManifoldPatch
    total_energy: chex.Array

def safe_cosine_similarity(v1, v2_b, eps=1e-8):
    """Robust cosine similarity that relies on JAX broadcasting."""
    n1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = jnp.linalg.norm(v2_b, axis=-1, keepdims=True)
    dot = jnp.sum(v1 * v2_b, axis=-1, keepdims=True)
    return jnp.where((n1 > eps) & (n2 > eps), dot / (n1 * n2 + eps), 0.0)


# --- MODEL MODULES (Corrected with NaN protection) ---

class EventProjection(nn.Module):
    config:Config; dtype:Any
    @nn.compact
    def __call__(self,event_canvas:Dict[str, chex.Array])->chex.Array:
        char_codes_i32=event_canvas['char_code'][...,0].astype(jnp.int32); c_emb=nn.Embed(300,64,dtype=self.dtype)(char_codes_i32%300)
        n_emb=nn.Embed(5,16,dtype=self.dtype)(event_canvas['num_bytes'][...,0])
        is_l_emb=nn.Embed(2,32,dtype=self.dtype,name="letter_embed")(event_canvas['is_letter'][...,0].astype(jnp.int32))
        is_n_emb=nn.Embed(2,32,dtype=self.dtype,name="number_embed")(event_canvas['is_number'][...,0].astype(jnp.int32))
        is_p_emb=nn.Embed(2,32,dtype=self.dtype,name="punct_embed")(event_canvas['is_punctuation'][...,0].astype(jnp.int32))
        is_w_emb=nn.Embed(2,32,dtype=self.dtype,name="space_embed")(event_canvas['is_whitespace'][...,0].astype(jnp.int32))
        props=jnp.concatenate([c_emb,n_emb,is_l_emb,is_n_emb,is_p_emb,is_w_emb],axis=-1)
        return nn.Dense(self.config.D_MODEL,dtype=self.dtype, param_dtype=jnp.float32)(props)

class GrammarFieldOperator(nn.Module):
    config:Config; dtype:Any
    @nn.compact
    def __call__(self,x:chex.Array)->chex.Array:
        cfg=self.config
        g=nn.softmax(nn.Dense(cfg.NUM_GRAMMAR_EXPERTS,use_bias=False,dtype=self.dtype,param_dtype=jnp.float32)(x).astype(jnp.float32),axis=-1)
        e=nn.Dense(cfg.NUM_GRAMMAR_EXPERTS*cfg.D_MODEL,dtype=self.dtype,param_dtype=jnp.float32)(x).reshape(*x.shape[:-1],cfg.NUM_GRAMMAR_EXPERTS,cfg.D_MODEL)
        return x+jnp.einsum('...k,...kd->...d',g,e)

class CognitiveObserver(nn.Module):
    config:Config; dtype:Any
    @nn.compact
    def __call__(self, hidden_states:chex.Array)->Tuple[chex.Array,chex.Array]:
        l=nn.Dense(self.config.D_MODEL,dtype=self.dtype,param_dtype=jnp.float32,name="cog_obs_head")(hidden_states)
        p=nn.softmax(l.astype(jnp.float32)); lp=nn.log_softmax(l.astype(jnp.float32)); e=-jnp.sum(p*lp,axis=-1)
        v=jnp.sum(p*jnp.power(lp+e[...,None],2),axis=-1); return e*v, jnp.sum(p*(lp-jnp.log(1./(self.config.PATCHES_PER_PAGE*self.config.D_MODEL))),axis=-1)

class CNNPerceiver(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, canvas: chex.Array) -> chex.Array:
        x=canvas
        for i, features in enumerate(self.config.CNN_FEATURES):
            x=nn.Conv(features,kernel_size=(3,3),strides=(2,2),name=f"conv_{i}",dtype=self.dtype,param_dtype=jnp.float32)(x)
            x=nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32, name=f"norm_{i}")(x); x=nn.gelu(x)
        B,H,W,D=x.shape; return x.reshape(B,H*W,D)

class TextDecoder(nn.Module):
    config: Config; dtype: Any
    @nn.compact
    def __call__(self, patch_remainders: chex.Array) -> chex.Array:
        cfg = self.config
        BP, N, D = patch_remainders.shape
        cnn_grid_h, cnn_grid_w = cfg.CANVAS_HEIGHT // 16, cfg.CANVAS_WIDTH // 16
        cnn_grid_features = cnn_grid_h * cnn_grid_w
        x = nn.Dense(cnn_grid_features * cfg.D_MODEL, dtype=self.dtype,param_dtype=jnp.float32, name="decoder_pre_proj")(patch_remainders.reshape(BP, -1))
        x = x.reshape(BP, cnn_grid_h, cnn_grid_w, cfg.D_MODEL) 
        deconv_features = reversed(cfg.CNN_FEATURES[:-1])
        for i, features in enumerate(deconv_features):
            x = nn.ConvTranspose(features=features, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name=f"deconv_{i}", dtype=self.dtype,param_dtype=jnp.float32)(x)
            x = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32, name=f"denorm_{i}")(x)
            x = nn.gelu(x)
        logits = nn.Dense(256, dtype=jnp.float32,param_dtype=jnp.float32, name="output_logits")(x)
        return jax.image.resize(logits, (BP, cfg.CANVAS_HEIGHT, cfg.CANVAS_WIDTH, 256), method='bilinear')   

class PhysicsCell(nn.Module):
    config:Config; dtype:Any
    @nn.compact
    def __call__(self, carry: WorldState, _) -> Tuple[WorldState, PhysicsStepOutput]:
        ws = carry
        cfg=self.config; eps=1e-9
        
        intra_page_influence = jnp.mean(ws.pages.position.remainders, axis=2, keepdims=True)
        inter_page_influence = jnp.mean(intra_page_influence, axis=1, keepdims=True)
        total_influence = intra_page_influence + inter_page_influence 
        
        fr = safe_cosine_similarity(ws.pages.orientation, total_influence)
        w = nn.softmax(fr.astype(jnp.float32), axis=2)
        f = w * total_influence

        current_pos = ws.pages.position.remainders
        current_ten = ws.pages.tension.remainders
        pu = current_pos + f
        ou = ws.pages.orientation + f
        
        if cfg.USE_GRAMMAR_FIELD:
            pu = GrammarFieldOperator(cfg, self.dtype, name="pos_grammar")(pu)
            ou = GrammarFieldOperator(cfg, self.dtype, name="ori_grammar")(ou)
        
        dp = nn.Dense(cfg.D_MODEL, name="pos_dyn", dtype=self.dtype,param_dtype=jnp.float32)(pu)
        do = nn.Dense(cfg.D_MODEL, name="ori_dyn", dtype=self.dtype,param_dtype=jnp.float32)(ou)
        dt = nn.Dense(cfg.D_MODEL, name="ten_dyn", dtype=self.dtype,param_dtype=jnp.float32)(current_ten + f)

        n_pos_state = update_odometer_state(ws.pages.position, dp)
        no = (ws.pages.orientation + do) / (jnp.linalg.norm(ws.pages.orientation + do, axis=-1, keepdims=True) + eps)
        n_ten_state = update_odometer_state(ws.pages.tension, dt)
        
        n_pages_patches = ManifoldPatch(position=n_pos_state, orientation=no, tension=n_ten_state)
        
        potential_energy = jnp.sum(1.0 - jnp.cos(n_pages_patches.tension.remainders), axis=(-1,-2))
        historical_energy = jnp.sum(jnp.abs(n_pages_patches.tension.quotients.astype(self.dtype)), axis=(-1,-2)) * 1e-3
        current_total_energy = potential_energy + historical_energy
        
        final_ws = WorldState(pages=n_pages_patches, total_energy=current_total_energy)
        scan_output = PhysicsStepOutput(pages=n_pages_patches, total_energy=current_total_energy)
        return final_ws, scan_output

class UnifiedPhysicsModel(nn.Module):
    config:Config; dtype:Any
    def setup(self):
        cfg = self.config
        self.event_projection = EventProjection(cfg, self.dtype)
        self.cnn_perceiver = CNNPerceiver(cfg, self.dtype)
        self.text_decoder = TextDecoder(cfg, self.dtype) 
        
        ScannedPhysicsEngine = nn.scan(
            PhysicsCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0,
            length=self.config.SIMULATION_STEPS
        )
        self.physics_engine = ScannedPhysicsEngine(self.config, self.dtype, name="PhysicsCellScanner")
        if cfg.USE_COGNITIVE_OBSERVER: self.cognitive_observer = CognitiveObserver(cfg, self.dtype)

    def process_pages(self, ec_pages: Dict[str, chex.Array]) -> chex.Array:
        B, P, H, W = ec_pages['char_code'].shape
        
        def _reshape_and_add_channel(x):
            return x.reshape(B*P, H, W, 1)

        ec_flat_ch = jax.tree_util.tree_map(_reshape_and_add_channel, ec_pages)

        projected_flat = self.event_projection(ec_flat_ch)
        perceived_flat = self.cnn_perceiver(projected_flat)
        
        _, N, D = perceived_flat.shape
        return perceived_flat.reshape(B, P, N, D)
        
    def __call__(self,ec:Dict[str, chex.Array],deterministic:bool=False):
        cfg = self.config
        
        perceived_pages = self.process_pages(ec)
        B, P, N, D = perceived_pages.shape
        
        key = self.make_rng('params')
        pos_key, ori_key = jax.random.split(key)
        init_shape = (B, P, cfg.PATCHES_PER_PAGE, D)

        init_pos_rem = jax.random.normal(pos_key, init_shape, dtype=self.dtype) * 0.01
        init_pos_quo = jnp.zeros_like(init_pos_rem, dtype=jnp.int32)
        init_pos = OdometerState(remainders=init_pos_rem, quotients=init_pos_quo)
        init_ori = jax.random.normal(ori_key, init_shape, dtype=self.dtype)
        init_ori /= (jnp.linalg.norm(init_ori, axis=-1, keepdims=True) + 1e-9)
        init_ten_rem = jnp.ones(init_shape,dtype=self.dtype)
        init_ten_quo = jnp.zeros_like(init_ten_rem, dtype=jnp.int32)
        init_ten = OdometerState(remainders=init_ten_rem, quotients=init_ten_quo)
        ip = ManifoldPatch(position=init_pos, orientation=init_ori, tension=init_ten)
        i_s_energy = jnp.sum(1.0 - jnp.cos(ip.tension.remainders), axis=(-1,-2))
        i_s = WorldState(pages=ip, total_energy=i_s_energy)
        
        final_carry, scan_traj_over_time = self.physics_engine(i_s, None)
        final_state = final_carry
        
        patch_traj = jax.tree_util.tree_map(
            lambda x: jnp.transpose(x, (1, 0, 2, 3, 4)) if x.ndim == 5 
            else (jnp.transpose(x, (1, 0, 2)) if x.ndim == 3 else jnp.transpose(x, (1, 0))), 
            scan_traj_over_time
        )
        
        traj_pos_remainders = patch_traj.pages.position.remainders
        B, S, P, N, D = traj_pos_remainders.shape
        traj_pos_rem_flat = traj_pos_remainders.reshape(B * S * P, N, D)
        
        all_logits_flat = self.text_decoder(traj_pos_rem_flat)
        
        _, H, W, C = all_logits_flat.shape
        traj_logits = all_logits_flat.reshape(B, S, P, H, W, C)
        
        final_state_pos_rem_flat = final_state.pages.position.remainders.reshape(B*P, N, D)
        mean_final_pos_per_page = jnp.mean(final_state_pos_rem_flat, axis=1)
        
        chaos, kl = jnp.array(0.0), jnp.array(0.0)
        if cfg.USE_COGNITIVE_OBSERVER:
            chaos, kl = self.cognitive_observer(mean_final_pos_per_page)
            chaos = chaos.reshape(B, P)
        
        return final_state, patch_traj, traj_logits, chaos



@struct.dataclass
class NoveltyEngineState: position_history: chex.Array
def init_novelty_engine(cfg: Config) -> NoveltyEngineState:
    hist_shape = (cfg.NOVELTY_HISTORY_LENGTH, cfg.BATCH_SIZE, cfg.NUM_PAGES, cfg.PATCHES_PER_PAGE, cfg.D_MODEL)
    return NoveltyEngineState(position_history=jnp.zeros(hist_shape, dtype=jnp.float32))
@partial(jit, static_argnames=('history_length',))
def calculate_novelty_loss(pages: ManifoldPatch, hist_state: NoveltyEngineState, history_length: int) -> chex.Array:
    hist=jax.lax.stop_gradient(hist_state.position_history)
    current_positions = pages.position.remainders
    def compute_sim(h): return jnp.mean(safe_cosine_similarity(current_positions,h))
    return jnp.mean(jax.vmap(compute_sim)(hist))
@jit
def update_novelty_history(hist_state: NoveltyEngineState, new_pages: ManifoldPatch) -> NoveltyEngineState:
    hist=jnp.roll(hist_state.position_history,1,0); 
    new_hist=hist.at[0].set(lax.stop_gradient(new_pages.position.remainders))
    return NoveltyEngineState(position_history=new_hist)
class CustomTrainState(train_state.TrainState): novelty_engine_state: NoveltyEngineState

def _create_physics_train_step(model: UnifiedPhysicsModel, config: Config):
    @jit
    def train_step_fn(state: CustomTrainState, event_pages: Dict[str, chex.Array], loss_weights: Dict[str, jnp.ndarray]):
        
        def loss_fn(params):
            final_state, patch_traj, traj_logits, chaos = model.apply(
                {'params': params}, event_pages, deterministic=False, rngs={'params': jax.random.PRNGKey(state.step)}
            )
            metrics = {}
            targets = event_pages['char_code']
            
            metrics['loss_reconstruction'] = optax.softmax_cross_entropy_with_integer_labels(
                traj_logits.reshape(-1, 256), 
                jnp.broadcast_to(targets[:, None, ...], traj_logits.shape[:-1]).flatten()
            ).mean()

            initial_pos = patch_traj.pages.position.remainders[:, 0]
            final_pos = final_state.pages.position.remainders
            metrics['loss_instability'] = jnp.mean(jnp.square(angular_distance(final_pos, initial_pos)))
            
            initial_quotients = patch_traj.pages.position.quotients[:, 0]
            final_quotients = final_state.pages.position.quotients
            metrics['loss_odometer_drift'] = jnp.mean(jnp.square((final_quotients - initial_quotients).astype(jnp.float32)))

            metrics['loss_spacetime'] = metrics['loss_reconstruction'] + metrics['loss_instability'] + metrics['loss_odometer_drift']
            
            initial_energy = patch_traj.total_energy[:, 0]
            final_energy = final_state.total_energy
            metrics['loss_dissipation'] = jnp.mean(nn.relu(final_energy - initial_energy * config.DISSIPATION_FACTOR))
            metrics['loss_free_energy'] = jnp.mean(final_energy - config.FREE_ENERGY_TEMP * chaos)
            
            if config.USE_NOVELTY_ENGINE: metrics['loss_novelty'] = jnp.nan_to_num(calculate_novelty_loss(final_state.pages,state.novelty_engine_state,config.NOVELTY_HISTORY_LENGTH))
            else: metrics['loss_novelty']=0.0
            
            total_loss = sum(loss_weights.get(name, 1.0) * metrics.get(f'loss_{name}', 0.0) for name in loss_weights)
            metrics['total_loss'] = total_loss
            
            return total_loss, (metrics, final_state.pages, traj_logits[:,-1])

        (loss, (metrics, final_pages, final_logits)), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        if config.USE_NOVELTY_ENGINE:
            state = state.replace(novelty_engine_state=update_novelty_history(state.novelty_engine_state, final_pages))
        metrics['grad_norm']=optax.global_norm(grads)
        return state, metrics, final_logits
    return train_step_fn








class InteractivityState:
    def __init__(self):
        self.lock=threading.Lock()
        self.shutdown_event=threading.Event()
        self.ui_messages=deque(maxlen=100) # Increased size for better logging
        self.force_save=False
        self.preview_enabled=True

    def get_and_reset_force_save(self) -> bool: 
        with self.lock:
            s=self.force_save
            self.force_save=False
            return s

    def set_shutdown(self):
        self.shutdown_event.set()

    def add_message(self, msg: str):
        with self.lock:
            self.ui_messages.append(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {msg}")

    def get_messages(self) -> List[str]:
        # This is not used in the final implementation, but kept for potential debugging
        with self.lock:
            msgs = list(self.ui_messages)
            self.ui_messages.clear()
            return msgs

class PhysicsTrainer:
    def __init__(self, config: Config):
        self.config=config; self.console=Console(); self.interactive_state=InteractivityState()
        self.log_messages=deque(maxlen=100)
        self.last_metrics={}
        self.steps_per_sec=0.0; self.live_preview_panel=None
        self.dtype = jnp.bfloat16 if self.config.USE_BFLOAT16 else jnp.float32
        self.param_count=0;
        self.data_dir=Path(self.config.DATA_DIR); self.raw_text_path=self.data_dir/self.config.RAW_TEXT_FILE
        self.data_dir.mkdir(exist_ok=True)
        self.interactive_state.add_message(f"--- Byte-level ingestion. Target: {self.raw_text_path} ---")
        
        self.pid_controller = PIDLambdaController(
            targets={'loss_spacetime': 0.8, 'loss_dissipation': 0.1, 'loss_free_energy': 0.0, 'loss_novelty': 0.5,},
            base_weights={'spacetime': 1.0, 'dissipation': 1.0, 'free_energy': 1e-4, 'novelty': 0.1,},
            gains={
                'loss_spacetime':(0.5,0.05,0.1), 'loss_dissipation': (0.5, 0.05, 0.1), 
                'loss_free_energy': (0.5, 0.05, 0.1), 'loss_novelty':(0.6,0.05,0.15),
            }
        )
        self.current_loss_weights = self.pid_controller.base_weights

    def data_loader_thread(self, data_iterator, data_queue: queue.Queue):
        """Producer thread: pulls from tf.data and puts into a queue."""
        while not self.interactive_state.shutdown_event.is_set():
            try:
                data = next(data_iterator)
                data_queue.put(data)
            except StopIteration:
                self.interactive_state.add_message("Data iterator finished.")
                break
            except Exception as e:
                self.interactive_state.add_message(f"❌ Data loader thread error: {e}")
                time.sleep(1)

    def listen_for_keys(self):
        s=self.interactive_state;
        if platform.system()=="Windows": import msvcrt
        else: import sys, tty, termios, select
        try:
            if platform.system()!="Windows": fd,old=sys.stdin.fileno(),termios.tcgetattr(sys.stdin.fileno()); tty.setcbreak(fd)
            s.add_message("--- Controls: [s] Save | [w] Preview | [q] Quit ---")
            while not s.shutdown_event.is_set():
                if (platform.system()=="Windows" and msvcrt.kbhit()) or (platform.system()!="Windows" and select.select([sys.stdin],[],[],0.05)[0]):
                    k=msvcrt.getch().decode('utf-8','ignore') if platform.system()=="Windows" else sys.stdin.read(1)
                    if k in ['q','\x03']: 
                        s.set_shutdown()
                        s.add_message("Shutdown requested by user.")
                        break
                    elif k=='s': 
                        with s.lock: s.force_save=True
                        s.add_message("Manual save requested.")
                    elif k=='w': 
                        with s.lock: s.preview_enabled=not s.preview_enabled
                        s.add_message(f"👁️ Preview: {'ON' if s.preview_enabled else 'OFF'}")
                else: time.sleep(0.05)
        except Exception: s.add_message("❌ TTY input disabled.")
        finally:
            if platform.system()!="Windows" and 'old' in locals(): termios.tcsetattr(fd,termios.TCSADRAIN,old)

    def shutdown(self, signum=None, frame=None):
        if not self.interactive_state.shutdown_event.is_set():
            self.interactive_state.add_message("Shutdown signal received."); self.interactive_state.set_shutdown();

    def _save_checkpoint(self, state, path):
        path.parent.mkdir(exist_ok=True); state_cpu=jax.device_get(state)
        state_cpu=jax.tree_util.tree_map(lambda x:x.astype(jnp.float32) if x.dtype==jnp.bfloat16 else x,state_cpu)
        path.write_bytes(serialization.to_bytes(state_cpu))
        self.interactive_state.add_message(f"💾 Checkpoint saved to [cyan]{path.name}[/cyan] (step {int(state.step)})")

    def _get_gpu_stats(self):
        try: 
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1e9:.2f}/{m.total/1e9:.2f} GB",f"{u.gpu}%"
        except Exception: return "N/A", "N/A"

    def _format_byte_canvas_to_panel(self, byte_canvas_dict: Dict[str, np.ndarray], title: str) -> Panel:
        """
        Formats a byte canvas into a Rich Panel, sanitizing and cropping it to a fixed
        display size to prevent any UI hopping.
        """
        # Define a stable, fixed size for the preview area.
        display_height = 32
        display_width = 80

        try:
            full_canvas = byte_canvas_dict['char_code']
            h, w = full_canvas.shape
            
            # Crop the canvas to the maximum display dimensions.
            cropped_h = min(h, display_height)
            cropped_w = min(w, display_width)
            cropped_canvas = full_canvas[:cropped_h, :cropped_w]
            
            # Process each line to sanitize and enforce fixed width.
            fixed_lines = []
            for row in cropped_canvas:
                # Decode the row of bytes into a string, replacing errors.
                raw_line_str = row.astype(np.uint8).tobytes().decode('utf-8', errors='replace')
                
                # Sanitize the string: replace non-printable characters with a placeholder ('.').
                # This prevents control codes from affecting terminal layout.
                sanitized_line = "".join(c if c.isprintable() else '.' for c in raw_line_str)
                
                # Enforce exact width: pad with spaces if too short, truncate if too long.
                # This is the key step to guarantee a stable panel size.
                fixed_width_line = sanitized_line.ljust(display_width)[:display_width]
                fixed_lines.append(fixed_width_line)
            
            # Join the perfectly formatted lines.
            text_content = "\n".join(fixed_lines)
            
        except Exception as e:
            text_content = f"Error formatting canvas: {e}"
            
        # Create a Text object with no_wrap=True, as we have manually handled all formatting.
        return Panel(Text(text_content, no_wrap=True), title=title, border_style="magenta")





    def _generate_layout(self) -> Layout:
        layout=Layout()
        layout.split(Layout(name="header",size=3),Layout(name="main",ratio=1),Layout(name="footer",size=3))
        layout["main"].split_row(Layout(name="left",ratio=1,minimum_size=50),Layout(name="right",ratio=2))
        layout["footer"].split(Layout(name="log", ratio=1, minimum_size=5), Layout(name="progress", size=1))
        return layout

    def train(self):
        signal.signal(signal.SIGINT, self.shutdown); signal.signal(signal.SIGTERM, self.shutdown)
        self.console.print("--- 🚀 [bold]Starting Orchestrated Genesis Engine Training[/bold] ---")
        self.console.print("--- [b purple]Physics Mode:[/b purple] Multi-Page, Toroidal Spacetime Engaged ---")

        cfg = self.config
        data_queue = queue.Queue(maxsize=4)
        
        try:
            if not self.raw_text_path.exists():
                raise FileNotFoundError(f"Data file not found: '{self.raw_text_path}'")
            
            page_size = cfg.CANVAS_HEIGHT * cfg.CANVAS_WIDTH
            book_size = page_size * cfg.NUM_PAGES
            file_size = self.raw_text_path.stat().st_size

            event_fields = list(CharacterEvent.__annotations__.keys())
            
            def book_generator():
                with open(self.raw_text_path, 'rb') as f:
                    while not self.interactive_state.shutdown_event.is_set():
                        start = np.random.randint(0, max(1, file_size - book_size))
                        f.seek(start)
                        chunk = np.frombuffer(f.read(book_size), dtype=np.uint8)
                        if len(chunk) < book_size: continue
                        
                        book_data = {name: [] for name in event_fields}
                        page_chunks = np.split(chunk, cfg.NUM_PAGES)
                        for page_chunk in page_chunks:
                            event_dict = deterministic_byte_parser_cpu(page_chunk, cfg.CANVAS_WIDTH, cfg.CANVAS_HEIGHT)
                            for name in event_fields:
                                book_data[name].append(event_dict[name])
                        yield {name: np.stack(pages) for name, pages in book_data.items()}

            output_signature = {name: tf.TensorSpec(shape=(cfg.NUM_PAGES, cfg.CANVAS_HEIGHT, cfg.CANVAS_WIDTH), dtype=tf.int64) for name in event_fields}
            
            dataset = tf.data.Dataset.from_generator(book_generator, output_signature=output_signature)
            dataset = dataset.batch(self.config.SUPER_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            data_iterator = dataset.as_numpy_iterator()

            loader_thread = threading.Thread(target=self.data_loader_thread, args=(data_iterator, data_queue), daemon=True)
            loader_thread.start()

            self.interactive_state.add_message("--- Initializing model architecture... ---")
            model = UnifiedPhysicsModel(self.config, self.dtype); key = jax.random.PRNGKey(42)
            self.interactive_state.add_message("--- Warming up data pipeline... ---")
            
            first_super_batch = data_queue.get(timeout=300)
            dummy_book = jax.tree_util.tree_map(lambda x: x[0:self.config.BATCH_SIZE], first_super_batch)
            self.interactive_state.add_message("--- ✅ First data batch received. ---")
            
            with self.console.status("[cyan]Initializing parameters..."):
                params = model.init({'params': key}, dummy_book, deterministic=True)['params']
            self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(params)); 
            self.interactive_state.add_message(f'--- ✅ Model initialized with [yellow]{self.param_count/1e6:.2f}M[/yellow] parameters. ---')
            
            optimizer = wubu_optimizer(self.config.WUBU_LR)
            state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=optimizer, novelty_engine_state=init_novelty_engine(self.config))
            ckpt_path = Path(self.config.CHECKPOINT_DIR) / f"{self.config.BASENAME}.ckpt"
            if not self.config.FRESH_START and ckpt_path.exists():
                self.interactive_state.add_message(f"--- Attempting to resume from [cyan]{ckpt_path}[/cyan] ---")
                try:
                    with ckpt_path.open('rb') as f: state=serialization.from_bytes(state,f.read())
                    self.interactive_state.add_message(f"--- ✅ Resumed from step {int(state.step)}. ---")
                except Exception as e:
                    self.interactive_state.add_message(f"⚠️ [yellow]Could not resume checkpoint: {e}[/yellow]. Starting fresh.")
            
            jitted_train_step = _create_physics_train_step(model, self.config)
            
            with self.console.status("[cyan]Compiling training step..."):
                first_batch = jax.tree_util.tree_map(lambda x: x[:self.config.BATCH_SIZE], first_super_batch)
                initial_weights = {k: jnp.array(v, dtype=self.dtype) for k, v in self.current_loss_weights.items()}
                state, _, _ = jitted_train_step(state, first_batch, initial_weights)
            self.interactive_state.add_message("--- ✅ JAX compilation complete. ---")

        except Exception as e:
            self.console.print("\n" + "="*80, style="bold red")
            self.console.print(f"🔥🔥🔥 A CATASTROPHIC ERROR OCCURRED DURING SETUP: {e} 🔥🔥🔥", style="bold red", justify="center")
            self.console.print_exception(show_locals=True)
            self.console.print("="*80, style="bold red")
            self.shutdown()
            return

        key_listener = threading.Thread(target=self.listen_for_keys, daemon=True); key_listener.start()
        global_step = int(state.step)
        
        layout = self._generate_layout()
        progress = Progress(TextColumn("{task.description}"), BarColumn(), "•", TextColumn("Step {task.completed}/{task.total}"), "•", TimeRemainingColumn(), transient=True)
        layout["progress"].update(progress)
        
        try:
            steps_per_epoch = (file_size // book_size) // (self.config.SUPER_BATCH_SIZE * self.config.BATCH_SIZE)
            total_steps = self.config.EPOCHS * steps_per_epoch if steps_per_epoch > 0 else self.config.EPOCHS
        except ZeroDivisionError:
            self.console.print("[bold red]ERROR: Not enough data for a single epoch.[/bold red]")
            total_steps = 1
        
        main_task = progress.add_task(f"[b]Epoch 1/{self.config.EPOCHS}[/]", total=total_steps, completed=global_step)
        self.interactive_state.add_message("--- ✅ Starting training loop. ---")
        
        with Live(layout, screen=True, vertical_overflow="crop", console=self.console, refresh_per_second=10) as live:
            try:
                while not self.interactive_state.shutdown_event.is_set() and global_step < total_steps:
                    try:
                        super_batch = data_queue.get(block=True, timeout=0.5)
                        
                        for i in range(super_batch[list(super_batch.keys())[0]].shape[0] // self.config.BATCH_SIZE):
                            if self.interactive_state.shutdown_event.is_set(): break
                            step_start_time = time.time()
                            loss_weights_for_step = {k: jnp.array(v, dtype=self.dtype) for k, v in self.current_loss_weights.items()}
                            batch = jax.tree_util.tree_map(lambda x: x[i * self.config.BATCH_SIZE:(i + 1) * self.config.BATCH_SIZE], super_batch)
                            
                            state, metrics, final_logits = jitted_train_step(state, batch, loss_weights_for_step)
                            
                            self.last_metrics = jax.device_get(metrics)
                            self.current_loss_weights = self.pid_controller(self.last_metrics)
                            self.steps_per_sec = 1.0 / (time.time() - step_start_time + 1e-9)
                            global_step += 1
                            
                            if self.interactive_state.preview_enabled and (global_step % self.config.PREVIEW_EVERY_N_STEPS == 0):
                                predicted_bytes = jnp.argmax(final_logits, axis=-1)
                                first_page_pred_canvas = {'char_code': predicted_bytes[0, 0]}
                                self.live_preview_panel = self._format_byte_canvas_to_panel(
                                    jax.device_get(first_page_pred_canvas), 
                                    f"🔥 Generated Page 1/{cfg.NUM_PAGES} (Step {global_step})"
                                )
                            if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and global_step % self.config.SAVE_EVERY == 0):
                                self._save_checkpoint(state, ckpt_path)

                    except queue.Empty:
                        pass # No data ready, just loop and update UI.

                    # --- UI UPDATE ---
                    with self.interactive_state.lock:
                        for msg in self.interactive_state.ui_messages: self.log_messages.append(msg)
                        self.interactive_state.ui_messages.clear()
                    
                    prec="[purple]BF16[/]" if self.config.USE_BFLOAT16 else "[dim]FP32[/]"
                    p_on="[on green]ON[/]" if self.interactive_state.preview_enabled else "[on red]OFF[/]"
                    header=f"🚀🧠 [b]WPE Orchestrated[/b] | Step: {global_step} | SPS: {self.steps_per_sec:.2f} | Preview (w): {p_on}"
                    layout["header"].update(Panel(Align.center(header),style="magenta",title=f"[dim]Params: {self.param_count/1e6:.2f}M | Prec: {prec}[/dim]",title_align="right"))
                    mem,util=self._get_gpu_stats()
                    stats_tbl=Table.grid(expand=True,padding=(0,1)); stats_tbl.add_column("dim",width=15); stats_tbl.add_column(justify="right")
                    stats_tbl.add_row("Steps/sec",f"[blue]{self.steps_per_sec:6.2f}[/] 🚀"); stats_tbl.add_row("Wubu LR",f"[green]{self.config.WUBU_LR:.2e}[/]"); stats_tbl.add_row("GPU Mem/Util",f"[yellow]{mem}[/] / [yellow]{util}[/]")
                    loss_tbl=Table.grid(expand=True,padding=(0,1)); loss_tbl.add_column("dim"); loss_tbl.add_column("bright_white",justify="right"); loss_tbl.add_column("cyan",justify="right")
                    loss_map={'spacetime':'Spacetime', 'reconstruction':'  Reconstruction', 'instability':'  Instability', 'odometer_drift':'  Odometer Drift','dissipation':'Dissipation', 'free_energy':'Free Energy', 'novelty':'Novelty',}
                    loss_tbl.add_row("[b]Component[/b]","[b]Value[/b]","[b]Weight (λ)[/b]")
                    for name, display in loss_map.items():
                        val=self.last_metrics.get(f'loss_{name}')
                        if val is not None and np.isfinite(val) and val != 0.0:
                            loss_tbl.add_row(display,f"{float(val):.4f}",f"{self.current_loss_weights.get(name, 0.0):.3f}")
                    loss_tbl.add_row("[b]Total Loss[/b]",f"[b]{float(self.last_metrics.get('total_loss',0)):.4f}[/]")
                    layout["left"].update(Align.center(Group(Panel(stats_tbl,title="[b]📊 Core Stats[/]"),Panel(loss_tbl,title="[b]⚡ Orchestrated Physics[/]",border_style="yellow"))))
                    if self.live_preview_panel and self.interactive_state.preview_enabled: layout["right"].update(self.live_preview_panel)
                    else: layout["right"].update(Panel(Align.center("..."),border_style="dim"))
                    layout["log"].update(Panel(Text("\n".join(self.log_messages)),title="[dim]Console[/dim]",border_style="dim"))
                    epoch = (global_step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1
                    progress.update(main_task, completed=global_step, description=f"[b]Epoch {epoch}/{self.config.EPOCHS}[/]")
            finally:
                self.shutdown()
                if 'live' in locals() and live.is_started: live.stop()
                self.console.print("\n--- Training loop terminated. Saving final state... ---")
                if 'state' in locals(): self._save_checkpoint(state, ckpt_path)
                if loader_thread.is_alive(): loader_thread.join(timeout=1)
                if key_listener.is_alive(): key_listener.join(timeout=1)




def main():
    parser=argparse.ArgumentParser(description="Wubu Physics Engine (Orchestrated)"); cfg=Config(); PhysicsTrainer(cfg).train()

if __name__ == "__main__":
    main()