# topological_sequence_model.py
#
# A robust, high-performance, multi-GPU JAX/Flax implementation and training
# framework for the TSM, upgraded with a rich TUI, Q-Learner for dynamic LR,
# and robust checkpointing.
#
# Author: Your Name Here (with inspiration from Wubumind)

import os
import sys
import time
import requests
import argparse
import pickle
import signal
from pathlib import Path
from functools import partial
from collections import deque
from dataclasses import dataclass, field, replace

# --- Environment and JAX Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.jax_utils import replicate, unreplicate
import chex

# --- Rich TUI & Monitoring Dependencies ---
try:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.align import Align
    from rich.table import Table
except ImportError:
    print("[FATAL] Required dependencies missing: tensorflow, tensorflow-datasets, rich. Please install them.")
    sys.exit(1)

NVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pass

jax.config.update("jax_threefry_partitionable", True)

# ==============================================================================
# 1. CORE MODEL ARCHITECTURE (Largely Unchanged)
# ==============================================================================
@jax.jit
def poincare_simulation(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
    delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
    real_part = jnp.cos(delta_f32 / 2)
    imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
    return real_part + 1j * imag_part

@dataclass
class TSMConfig:
    vocab_size: int = 65; n_embd: int = 384; n_head: int = 6; n_layer: int = 6
    compression_factor: int = 4; max_seq_len: int = 256; dtype: jnp.dtype = jnp.float32

class HamiltonianEncoder(nn.Module):
    config: TSMConfig
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_conv_input = x[None, :, :]
        num_outputs = self.config.n_head * 3
        physics_params_flat = nn.Conv(
            features=num_outputs, kernel_size=(self.config.compression_factor,),
            strides=(self.config.compression_factor,), padding='VALID', dtype=self.config.dtype
        )(x_conv_input)
        physics_params_unbatched = physics_params_flat[0]
        compressed_len = physics_params_unbatched.shape[0]
        physics_params = physics_params_unbatched.reshape(compressed_len, self.config.n_head, 3)
        delta = nn.tanh(physics_params[..., 0]) * jnp.pi
        chi = nn.tanh(physics_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(physics_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta, chi, radius], axis=-1)

class ImplicitAttention(nn.Module):
    config: TSMConfig
    @nn.compact
    def __call__(self, x: jnp.ndarray, tsv: jnp.ndarray) -> jnp.ndarray:
        seq_len, n_embd = x.shape
        v_proj = nn.Dense(n_embd, name="v_proj", dtype=self.config.dtype)(x)
        v_compressed = nn.avg_pool(v_proj[None, :, :], window_shape=(self.config.compression_factor,), strides=(self.config.compression_factor,), padding='VALID')[0]
        tsv_flat = tsv.reshape((tsv.shape[0], -1))
        tsv_batched, v_batched = tsv_flat[None, :, :], v_compressed[None, :, :]
        interp_params_batched = nn.ConvTranspose(
            features=self.config.n_head * 3, kernel_size=(self.config.compression_factor,),
            strides=(self.config.compression_factor,), padding='VALID', name="upsample_params", dtype=self.config.dtype
        )(tsv_batched)
        interp_v_batched = nn.ConvTranspose(
            features=n_embd, kernel_size=(self.config.compression_factor,),
            strides=(self.config.compression_factor,), padding='VALID', name="upsample_v", dtype=self.config.dtype
        )(v_batched)
        interp_params_flat, interp_v = interp_params_batched[0][:seq_len, :], interp_v_batched[0][:seq_len, :]
        interp_params = interp_params_flat.reshape(seq_len, self.config.n_head, 3)
        delta, chi = interp_params[..., 0], interp_params[..., 1]
        t_co = poincare_simulation(delta, chi)
        head_dim = n_embd // self.config.n_head
        interp_v_headed = interp_v.reshape(seq_len, self.config.n_head, head_dim)
        real_mod, imag_mod = jnp.real(t_co)[:, :, None] * interp_v_headed, jnp.imag(t_co)[:, :, None] * interp_v_headed
        attended_v = jnp.concatenate([real_mod, imag_mod], axis=-1).reshape(seq_len, -1)
        return nn.Dense(n_embd, name="output_proj", dtype=self.config.dtype)(attended_v)

class _TSMBlockLogic(nn.Module):
    config: TSMConfig
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h_norm1 = nn.LayerNorm(name="ln_1", dtype=self.config.dtype)(x)
        tsv = HamiltonianEncoder(config=self.config, name="hamiltonian_encoder")(h_norm1)
        attention_output = ImplicitAttention(config=self.config, name="implicit_attention")(h_norm1, tsv)
        x = x + attention_output
        h_norm2 = nn.LayerNorm(name="ln_2", dtype=self.config.dtype)(x)
        mlp_output = nn.Dense(4 * self.config.n_embd, name="mlp_fc", dtype=self.config.dtype)(h_norm2)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dense(self.config.n_embd, name="mlp_proj", dtype=self.config.dtype)(mlp_output)
        return x + mlp_output

class TSMBlock(nn.Module):
    config: TSMConfig
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        VmappedLogic = nn.vmap(
            _TSMBlockLogic, in_axes=0, out_axes=0,
            variable_axes={'params': None}, split_rngs={'params': False}
        )
        return VmappedLogic(config=self.config, name="VmappedBlock")(x)

class TSM_GPT2(nn.Module):
    config: TSMConfig
    @nn.compact
    def __call__(self, idx: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        seq_len = idx.shape[1]
        token_emb = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embd, name="wte", dtype=self.config.dtype)(idx)
        pos = jnp.arange(0, seq_len)
        pos_emb = nn.Embed(num_embeddings=self.config.max_seq_len, features=self.config.n_embd, name="wpe", dtype=self.config.dtype)(pos)
        x = token_emb + pos_emb
        x = nn.Dropout(rate=0.1)(x, deterministic=deterministic)
        for i in range(self.config.n_layer):
            x = TSMBlock(config=self.config, name=f"block_{i}")(x)
        x = nn.LayerNorm(name="ln_f", dtype=self.config.dtype)(x)
        return nn.Dense(self.config.vocab_size, name="lm_head", dtype=self.config.dtype)(x)

# ==============================================================================
# 2. ADVANCED TRAINING TOOLKIT
# ==============================================================================
Q_CONTROLLER_CONFIG_TSM = {
    "q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1],
    "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 1e-3,
    "metric_history_len": 200, "loss_min": 0.1, "loss_max": 5.0, "exploration_rate_q": 0.3,
    "min_exploration_rate": 0.05, "exploration_decay": 0.9998, "trend_window": 100,
    "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": -2.0,
    "stagnation_penalty": -0.5, "warmup_steps": 200, "warmup_lr_start": 1e-6
}

@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class QControllerState:
    q_table: chex.Array; metric_history: chex.Array; trend_history: chex.Array
    current_value: jnp.ndarray; exploration_rate: jnp.ndarray; step_count: jnp.ndarray
    last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray
    def tree_flatten(self): return (self.q_table, self.metric_history, self.trend_history, self.current_value, self.exploration_rate, self.step_count, self.last_action_idx, self.last_reward, self.status_code), None
    @classmethod
    def tree_unflatten(cls, aux_data, children): return cls(*children)

def init_q_controller(config):
    return QControllerState(
        q_table=jnp.zeros((config["q_table_size"], config["num_lr_actions"]), dtype=jnp.float32),
        metric_history=jnp.full((config["metric_history_len"],), (config["loss_min"] + config["loss_max"]) / 2, dtype=jnp.float32),
        trend_history=jnp.zeros((config["trend_window"],), dtype=jnp.float32),
        current_value=jnp.array(config["warmup_lr_start"], dtype=jnp.float32),
        exploration_rate=jnp.array(config["exploration_rate_q"], dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
        status_code=jnp.array(0, dtype=jnp.int32)
    )

@jax.jit
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey):
    config = Q_CONTROLLER_CONFIG_TSM
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]
        new_value = config["warmup_lr_start"] * (1 - alpha) + config["lr_max"] * 0.5 * alpha
        return replace(state, current_value=new_value, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - 5, 5))
        state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        explore_key, action_key = jax.random.split(key)
        def explore(): return jax.random.randint(action_key, (), 0, config["num_lr_actions"])
        def exploit(): return jnp.argmax(state.q_table[state_idx])
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate, explore, exploit)
        selected_factor = jnp.array(config["lr_change_factors"])[action_idx]
        new_value = jnp.clip(state.current_value * selected_factor, config["lr_min"], config["lr_max"])
        return replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)

@jax.jit
def q_controller_update(state: QControllerState, metric_value: float):
    config = Q_CONTROLLER_CONFIG_TSM
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    new_trend_history = jnp.roll(state.trend_history, -1).at[-1].set(metric_value)
    def perform_update(st):
        x = jnp.arange(config["trend_window"], dtype=jnp.float32)
        y = new_trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(
            slope < -config["improve_threshold"], lambda: (jnp.array(1), 1.0),
            lambda: jax.lax.cond(
                slope > config["regress_threshold"], lambda: (jnp.array(3), config["regress_penalty"]),
                lambda: (jnp.array(2), config["stagnation_penalty"])
            )
        )
        old_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config["metric_history_len"]-5, 5))
        last_state_idx = jnp.clip(((old_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(new_metric_history, config["metric_history_len"]-5, 5))
        next_state_idx = jnp.clip(((new_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        current_q = st.q_table[last_state_idx, st.last_action_idx]
        max_next_q = jnp.max(st.q_table[next_state_idx])
        new_q = current_q + config["learning_rate_q"] * (reward + config["discount_factor_q"] * max_next_q - current_q)
        new_q_table = st.q_table.at[last_state_idx, st.last_action_idx].set(new_q)
        new_exp_rate = jnp.maximum(config["min_exploration_rate"], st.exploration_rate * config["exploration_decay"])
        return replace(st, q_table=new_q_table, exploration_rate=new_exp_rate, last_reward=reward, status_code=status_code)
    can_update = (state.step_count > config["warmup_steps"]) & (state.step_count > config["trend_window"]) & (state.last_action_idx >= 0)
    new_state = jax.lax.cond(can_update, perform_update, lambda s: s, state)
    return replace(new_state, metric_history=new_metric_history, trend_history=new_trend_history)

class TSMTrainState(train_state.TrainState):
    q_state: QControllerState

# ==============================================================================
# 3. DATA HANDLING
# ==============================================================================
def create_dataset(data: np.ndarray, batch_size: int, block_size: int):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(block_size + 1, shift=1, drop_remainder=True).flat_map(lambda w: w.batch(block_size + 1))
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset

# ==============================================================================
# 4. TRAINING FRAMEWORK
# ==============================================================================
class TSMTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.console = Console()
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._shutdown_handler)
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.num_devices = jax.local_device_count()
        self.global_batch_size = self.args.batch_size
        self.batch_size_per_device = self.global_batch_size // self.num_devices
        if self.global_batch_size % self.num_devices != 0:
            sys.exit(f"[FATAL] Global batch size ({self.global_batch_size}) must be divisible by num devices ({self.num_devices}).")
        self.last_metrics_for_ui = {}
        self.hist_len = 200
        self.loss_hist = deque(maxlen=self.hist_len); self.val_loss_hist = deque(maxlen=self.hist_len); self.lr_hist = deque(maxlen=self.hist_len)

    def _shutdown_handler(self, signum, frame):
        self.console.print("\n[bold yellow]! Ctrl+C detected. Initiating graceful shutdown...[/bold yellow]")
        self.should_shutdown = True

    def _get_gpu_stats(self):
        if not NVML_AVAILABLE: return "N/A", "N/A"
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0)
            m=pynvml.nvmlDeviceGetMemoryInfo(h)
            u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception: return "N/A", "N/A"

    def _get_sparkline(self, data: deque, w=30):
        s=" ▂▃▄▅▆▇█"
        hist=np.array(list(data))
        if len(hist)<2: return " "*w
        hist=hist[-w:]
        min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s))
        indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def _create_layout(self) -> Layout:
        root_layout = Layout(name="root")
        root_layout.split_column(Layout(name="header", size=3), Layout(name="main", ratio=1), Layout(self.progress, name="footer", size=3))
        root_layout["main"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
        precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
        header_text = f"🧬 [bold]Topological Sequence Model[/] | Basename: [cyan]{self.args.basename}[/] | Devices: {self.num_devices} | Precision: {precision_str}"
        root_layout["header"].update(Panel(Align.center(header_text), style="bold blue"))
        loss = self.last_metrics_for_ui.get('loss', 0); val_loss = self.last_metrics_for_ui.get('val_loss', 0)
        stats_tbl = Table.grid(expand=True); stats_tbl.add_column(style="dim",width=14); stats_tbl.add_column()
        stats_tbl.add_row("Train Loss", f"[cyan]{loss:.4f}[/]"); stats_tbl.add_row("Val Loss", f"[magenta]{val_loss:.4f}[/]")
        mem, util = self._get_gpu_stats(); stats_tbl.add_row("GPU Mem / Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
        left_panel_group = [Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue")]
        q_status_code = int(self.last_metrics_for_ui.get('q_status_code', 0))
        status_map = {0: ("WARMUP", "blue", "🐣"), 1: ("IMPROVING", "green", "😎"), 2: ("STAGNATED", "yellow", "🤔"), 3: ("REGRESSING", "red", "😠")}
        q_status_str, q_color, q_emoji = status_map.get(q_status_code, ("N/A", "dim", "🤖"))
        q_tbl = Table.grid(expand=True); q_tbl.add_column(style="dim",width=12); q_tbl.add_column()
        q_tbl.add_row("Status", f"[{q_color}]{q_status_str}[/{q_color}] {q_emoji}");
        q_tbl.add_row("Current LR", f"[{q_color}]{self.last_metrics_for_ui.get('lr', 0.0):.2e}[/]")
        q_tbl.add_row("Reward", f"{self.last_metrics_for_ui.get('q_reward', 0.0):+.2f}")
        left_panel_group.append(Panel(q_tbl, title="[bold]🤖 Q-Controller (LR)[/]", border_style="green"))
        root_layout["left"].update(Align.center(Panel(Align.center(Group(*left_panel_group)))))
        spark_w = 40
        graphs = [
            Panel(Align.center(f"{self._get_sparkline(self.loss_hist, spark_w)}"), title="Train Loss", height=3, border_style="cyan"),
            Panel(Align.center(f"{self._get_sparkline(self.val_loss_hist, spark_w)}"), title="Val Loss", height=3, border_style="magenta"),
            Panel(Align.center(f"{self._get_sparkline(self.lr_hist, spark_w)}"), title="Learning Rate", height=3, border_style="yellow"),
        ]
        root_layout["right"].update(Panel(Group(*graphs), title="[bold]📈 Live Trends[/]"))
        return root_layout

    def train(self):
        self.console.print("--- 🧠 TSM Training Initializing ---", style="bold yellow")
        data_path = self.args.data_path
        if not os.path.exists(data_path):
            self.console.print(f"Data file not found at {data_path}. Downloading tiny-shakespeare...", style="yellow")
            with open(data_path, 'w') as f: f.write(requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text)
        with open(data_path, 'r') as f: text = f.read()
        chars = sorted(list(set(text))); vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}; itos = {i: ch for i, ch in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.uint16)
        train_data, val_data = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]
        train_ds = create_dataset(train_data, self.global_batch_size, self.args.block_size)
        val_ds = create_dataset(val_data, self.global_batch_size, self.args.block_size)
        self.console.print(f"✅ Data loaded: {len(text):,} characters, {vocab_size} unique.")
        config = TSMConfig(vocab_size=vocab_size, max_seq_len=self.args.block_size, dtype=self.dtype)
        model = TSM_GPT2(config)
        key = jax.random.PRNGKey(self.args.seed)
        key, init_key = jax.random.split(key)
        params = model.init(init_key, jnp.zeros((1, self.args.block_size), dtype=np.uint16), deterministic=True)['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        self.console.print(f"✅ Model initialized. Params: {param_count:,}")
        optimizer = optax.adamw(learning_rate=self.args.learning_rate)
        state = TSMTrainState.create(apply_fn=model.apply, params=params, tx=optimizer, q_state=init_q_controller(Q_CONTROLLER_CONFIG_TSM))
        ckpt_dir = Path(f"./checkpoints/{self.args.basename}").resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path_best = Path(f"{self.args.basename}_best.pkl")
        best_val_loss, start_epoch, global_step = float('inf'), 0, 0
        state = checkpoints.restore_checkpoint(ckpt_dir, target=state)
        meta_path = ckpt_dir / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                start_epoch = meta.get('epoch', 0) + 1; global_step = meta.get('global_step', 0); best_val_loss = meta.get('best_val_loss', float('inf'))
                self.console.print(f"--- Resuming training from epoch {start_epoch}, step {global_step} ---")
        def loss_fn_logic(params, batch, dropout_key, apply_fn):
            x, y = batch[:, :-1], batch[:, 1:]
            logits = apply_fn({'params': params}, x, deterministic=False, rngs={'dropout': dropout_key})
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        def train_step_logic(state, batch, key):
            q_key, dropout_key = jax.random.split(key)
            new_q_state = q_controller_choose_action(state.q_state, q_key)
            learning_rate = new_q_state.current_value
            loss, grads = jax.value_and_grad(loss_fn_logic)(state.params, batch, dropout_key, state.apply_fn)
            tx = optax.adamw(learning_rate=learning_rate)
            updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            final_q_state = q_controller_update(new_q_state, loss)
            metrics = {'loss': loss, 'lr': learning_rate, 'q_status_code': final_q_state.status_code, 'q_reward': final_q_state.last_reward}
            new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, q_state=final_q_state)
            return new_state, metrics
        if self.num_devices > 1:
            state = replicate(state)
            @partial(jax.pmap, axis_name='batch')
            def pmapped_train_step(state, batch, key):
                state, metrics = train_step_logic(state, batch, key)
                metrics = jax.lax.pmean(metrics, axis_name='batch')
                return state, metrics
            jitted_train_step = pmapped_train_step
        else:
            jitted_train_step = jax.jit(train_step_logic)
        @partial(jax.jit, static_argnames=('apply_fn',))
        def eval_step(params, batch, apply_fn):
            x, y = batch[:, :-1], batch[:, 1:]
            logits = apply_fn({'params': params}, x, deterministic=True)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        self.console.print("[bold yellow]🚀 JIT compiling training step...[/bold yellow]")
        dummy_batch = next(iter(tfds.as_numpy(train_ds)))
        if self.num_devices > 1:
            dummy_batch = jax.tree_util.tree_map(lambda x: x.reshape((self.num_devices, self.batch_size_per_device) + x.shape[1:]), dummy_batch)
            key, *step_keys = jax.random.split(key, self.num_devices + 1)
            step_keys = jnp.stack(step_keys)
        else:
            key, step_keys = jax.random.split(key)
        jitted_train_step(state, dummy_batch, step_keys)
        self.console.print("[green]✅ Compilation complete.[/green]")
        steps_per_epoch = len(train_data) // self.global_batch_size
        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.4f}[/]"), BarColumn(), "•", TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"), "•", TimeRemainingColumn(), "•", TextColumn("Ctrl+C to Save & Exit"))
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=best_val_loss, step=0, steps_per_epoch=steps_per_epoch)
        epoch = 0
        try:
            with Live(self._create_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for epoch in range(start_epoch, self.args.epochs):
                    if self.should_shutdown: break
                    train_iterator = tfds.as_numpy(train_ds)
                    for step_in_epoch, batch in enumerate(train_iterator):
                        if self.should_shutdown: break
                        if self.num_devices > 1:
                            batch = jax.tree_util.tree_map(lambda x: x.reshape((self.num_devices, self.batch_size_per_device) + x.shape[1:]), batch)
                            key, *step_keys = jax.random.split(key, self.num_devices + 1)
                            step_keys = jnp.stack(step_keys)
                        else:
                            key, step_keys = jax.random.split(key)
                        state, metrics = jitted_train_step(state, batch, step_keys)
                        metrics_cpu = jax.device_get(unreplicate(metrics)) if self.num_devices > 1 else jax.device_get(metrics)
                        self.last_metrics_for_ui.update({k: v.item() for k, v in metrics_cpu.items()})
                        self.loss_hist.append(self.last_metrics_for_ui['loss']); self.lr_hist.append(self.last_metrics_for_ui['lr'])
                        global_step += 1
                        self.progress.update(epoch_task, step=step_in_epoch + 1)
                        if global_step % self.args.eval_interval == 0:
                            eval_state = unreplicate(state) if self.num_devices > 1 else state
                            val_losses = []
                            val_iterator = iter(tfds.as_numpy(val_ds))
                            for _ in range(self.args.eval_batches):
                                try:
                                    val_batch = next(val_iterator)
                                    loss_val = eval_step(eval_state.params, val_batch, eval_state.apply_fn)
                                    val_losses.append(loss_val)
                                except StopIteration: break
                            val_loss = np.mean([v.item() for v in val_losses]) if val_losses else float('inf')
                            self.last_metrics_for_ui['val_loss'] = val_loss
                            self.val_loss_hist.append(val_loss)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                self.console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.4f} @ step {global_step}. Saving...[/bold magenta]")
                                best_ckpt_data = {'params': jax.device_get(eval_state.params), 'config': config}
                                with open(ckpt_path_best, 'wb') as f: pickle.dump(best_ckpt_data, f)
                            self.progress.update(epoch_task, val_loss=best_val_loss)
                        live.update(self._create_layout())
                    self.progress.update(epoch_task, advance=1, step=0)
                    checkpoints.save_checkpoint(ckpt_dir, target=state, step=epoch, overwrite=True, keep=3)
                    meta = {'epoch': epoch, 'global_step': global_step, 'best_val_loss': best_val_loss}
                    with open(meta_path, 'wb') as f: pickle.dump(meta, f)
        finally:
            self.console.print(f"\n[yellow]--- Training loop exited. ---[/yellow]")
            if 'state' in locals():
                self.console.print(f"\n[yellow]--- Saving final state... ---[/yellow]")
                checkpoints.save_checkpoint(ckpt_dir, target=state, step=epoch, overwrite=True, keep=3)
                meta = {'epoch': epoch, 'global_step': global_step, 'best_val_loss': best_val_loss}
                with open(meta_path, 'wb') as f: pickle.dump(meta, f)
                self.console.print(f"✅ Final resume-state saved to [green]{ckpt_dir}[/green]")
            config_path = Path(f"{self.args.basename}_config.pkl")
            with open(config_path, 'wb') as f:
                pickle.dump({'config': config, 'stoi': stoi, 'itos': itos}, f)
            self.console.print(f"✅ Model config and vocab saved to [green]{config_path}[/green]")
            if ckpt_path_best.exists():
                self.console.print(f"👑 Best model (by validation) is at [bold magenta]{ckpt_path_best}[/bold magenta]")

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="TSM Training & Generation Framework", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_train = subparsers.add_parser("train", help="Train the TSM model.")
    p_train.add_argument('--basename', type=str, required=True, help="Base name for saving model files (e.g., 'tsm_shakespeare').")
    p_train.add_argument('--data-path', type=str, default="input.txt", help="Path to the training data text file.")
    p_train.add_argument('--epochs', type=int, default=10, help="Total number of epochs to train for.")
    p_train.add_argument('--batch-size', type=int, default=64, help="Global batch size across all devices.")
    p_train.add_argument('--block-size', type=int, default=256, help="Sequence length for the model.")
    p_train.add_argument('--learning-rate', type=float, default=3e-4, help="Initial learning rate (will be controlled by Q-learner).")
    p_train.add_argument('--eval-interval', type=int, default=250, help="Evaluate on validation set every N steps.")
    p_train.add_argument('--eval-batches', type=int, default=50, help="Number of batches to use for validation loss estimation.")
    p_train.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision for training.")
    p_train.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p_gen = subparsers.add_parser("generate", help="Generate text from a trained TSM model.")
    p_gen.add_argument('--basename', type=str, required=True, help="Basename of the model to load (e.g., 'tsm_shakespeare').")
    p_gen.add_argument('--prime-text', type=str, default="JULIET:\n", help="The starting text to prime the model.")
    p_gen.add_argument('--max-new-tokens', type=int, default=1000, help="Number of new tokens to generate.")
    p_gen.add_argument('--seed', type=int, default=int(time.time()), help="Random seed for generation.")
    args = parser.parse_args()

    if args.command == "train":
        TSMTrainer(args).train()
    elif args.command == "generate":
        console = Console()
        config_path = Path(f"{args.basename}_config.pkl")
        ckpt_path_best = Path(f"{args.basename}_best.pkl")
        if not config_path.exists() or not ckpt_path_best.exists():
            console.print(f"[bold red]Error: Cannot find model files.[/bold red] Expected config at '{config_path}' and best checkpoint at '{ckpt_path_best}'.")
            sys.exit(1)
        console.print(f"--- Loading model [cyan]{args.basename}[/cyan] for generation ---")
        with open(config_path, 'rb') as f: data = pickle.load(f)
        config, stoi, itos = data['config'], data['stoi'], data['itos']
        with open(ckpt_path_best, 'rb') as f: ckpt_data = pickle.load(f)
        params = ckpt_data['params']
        model = TSM_GPT2(config)
        
        @partial(jax.jit, static_argnames=('max_new_tokens', 'block_size'))
        def _generate_jit(params, prime_tokens, max_new_tokens, block_size, key):
            
            def scan_body(carry, _):
                key, window = carry
                key, subkey = jax.random.split(key)

                logits = model.apply({'params': params}, window, deterministic=True)
                idx_next = jax.random.categorical(subkey, logits[:, -1, :])
                
                # --- [FIX] Cast to uint16 to match carry dtype ---
                idx_next = idx_next.astype(jnp.uint16)

                # Slide the window
                new_window = jnp.roll(window, shift=-1, axis=1).at[:, -1].set(idx_next)

                return (key, new_window), idx_next

            # Initialize the window
            prime_len = prime_tokens.shape[1]
            padded_window = jnp.zeros((1, block_size), dtype=jnp.uint16)
            initial_window = padded_window.at[:, -prime_len:].set(prime_tokens)
            
            initial_carry = (key, initial_window)
            
            _, all_new_tokens = jax.lax.scan(scan_body, initial_carry, None, length=max_new_tokens)

            return jnp.concatenate([prime_tokens, all_new_tokens.reshape(1, -1)], axis=1)

        prime_tokens = jnp.array([[stoi.get(c, 0) for c in args.prime_text]], dtype=jnp.uint16)
        key = jax.random.PRNGKey(args.seed)
        
        console.print("[yellow]JIT Compiling generation function...[/yellow]")
        generated_tokens = _generate_jit(
            params, 
            prime_tokens, 
            args.max_new_tokens, 
            config.max_seq_len, 
            key
        )[0]
        
        console.print("--- [bold green]Generated Text[/bold green] ---")
        print("".join([itos.get(i, '?') for i in generated_tokens.tolist()]))

if __name__ == "__main__":
    main()