class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray
@dataclass(frozen=True)
class QControllerConfig:
    num_lr_actions: int = 5; lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; lr_min: float = 1e-4; lr_max: float = 5e-2
    metric_history_len: int = 100; exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    warmup_steps: int = 500; warmup_lr_start: float = 1e-6
def init_q_controller(config: QControllerConfig) -> QControllerState:
    return QControllerState(q_table=jnp.zeros(config.num_lr_actions), metric_history=jnp.zeros(config.metric_history_len),
                            current_lr=jnp.array(config.warmup_lr_start), exploration_rate=jnp.array(config.exploration_rate_q),
                            step_count=jnp.array(0), last_action_idx=jnp.array(-1, dtype=jnp.int32), status_code=jnp.array(0, dtype=jnp.int32))
@partial(jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey, config: QControllerConfig, target_lr: float) -> QControllerState:
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0, dtype=jnp.int32), last_action_idx=jnp.array(-1, dtype=jnp.int32))
    def regular_action():
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < jnp.squeeze(state.exploration_rate),
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions, dtype=jnp.int32),
            lambda: jnp.argmax(state.q_table).astype(jnp.int32))
        new_lr = jnp.clip(state.current_lr * jnp.array(config.lr_change_factors)[action_idx], config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx, status_code=jnp.array(0, dtype=jnp.int32))
    return jax.lax.cond(jnp.squeeze(state.step_count) < config.warmup_steps, warmup_action, regular_action)
@partial(jit, static_argnames=('config',))
def q_controller_update(state: QControllerState, metric_value: chex.Array, config: QControllerConfig) -> QControllerState:
    new_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    st = state.replace(metric_history=new_history)
    def perform_q_update(s: QControllerState) -> QControllerState:
        reward = -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 10, 10))
        is_improving = reward > -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 20, 10))
        status = jax.lax.select(is_improving, jnp.array(1, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32))
        old_q = s.q_table[s.last_action_idx]
        new_q = old_q + config.learning_rate_q * (reward - old_q)
        return s.replace(q_table=s.q_table.at[s.last_action_idx].set(new_q),
                         exploration_rate=jnp.maximum(config.min_exploration_rate, s.exploration_rate * config.exploration_decay),
                         status_code=status)
    can_update = (jnp.squeeze(st.step_count) > config.warmup_steps) & (jnp.squeeze(st.last_action_idx) >= 0)
    return jax.lax.cond(can_update, perform_q_update, lambda s: s, st)
class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: QControllerState; opt_state: optax.OptState

