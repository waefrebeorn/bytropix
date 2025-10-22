import time
import numpy as np
import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Any

# JAX can be quite verbose with warnings; this helps keep the demo output clean.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# SCRIPT OVERVIEW
# ==============================================================================
# This script demonstrates two advanced, dynamic controllers for deep learning:
# 1. A PID Controller: Used for automatically balancing multiple loss terms.
# 2. A Q-Learning Controller: An agent that learns an optimal learning rate schedule.
# Both are presented as standalone, understandable components with a simulation
# to show them in action.
# ==============================================================================


# ==============================================================================
# 1. PID CONTROLLER (For dynamically weighting loss terms)
# ==============================================================================
# PURPOSE: In complex models, you might have multiple loss functions (e.g., a
# reconstruction loss and a regularization loss). Manually tuning their relative
# weights is difficult. A PID controller automates this by treating each loss
# value as a system to be controlled, trying to "steer" it towards a target value
# by adjusting its weight. It's like a thermostat for your loss functions.

class PIDLambdaController:
    """
    A Proportional-Integral-Derivative (PID) controller to dynamically adjust
    loss weights (lambdas) based on performance metrics.
    """
    def __init__(self,
                 targets: Dict[str, float],
                 base_weights: Dict[str, float],
                 gains: Dict[str, Tuple[float, float, float]]):
        """
        Initializes the PID controller.

        Args:
            targets: A dictionary mapping metric names to their desired target values.
                     The controller will try to make the metric's value equal this target.
            base_weights: The initial/default weight for each metric's loss term.
            gains: A dictionary mapping metric names to their (Kp, Ki, Kd) gains.
                   These are the tuning "knobs" of the PID controller:
                   - Kp (Proportional): Reacts to the current error.
                   - Ki (Integral): Reacts to the sum of past errors.
                   - Kd (Derivative): Reacts to the rate of change of the error.
        """
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        # Internal state for tracking errors over time for the I and D terms.
        self.state = {
            'integral_error': {k: 0.0 for k in targets.keys()},
            'last_error': {k: 0.0 for k in targets.keys()}
        }

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculates the new set of loss weights based on the latest metrics."""
        final_lambdas = {}
        for name, base_weight in self.base_weights.items():
            final_lambdas[name] = float(base_weight)  # Start with the base weight

            # Only adjust weights for metrics being actively controlled.
            if name in self.targets:
                current_loss = last_metrics.get(name)
                if current_loss is None:
                    continue

                # --- Core PID Calculation ---
                kp, ki, kd = self.gains[name]
                target = self.targets[name]
                error = float(current_loss) - target

                # Proportional term: How far are we from the target right now?
                p_term = kp * error

                # Integral term: What is the accumulated error over time?
                # This helps eliminate steady-state error.
                self.state['integral_error'][name] += error
                # "Anti-windup": Clamp the integral term to prevent it from growing
                # infinitely large and causing massive overshoots.
                self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                i_term = ki * self.state['integral_error'][name]

                # Derivative term: How fast is the error changing?
                # This helps dampen oscillations and predict future error.
                derivative = error - self.state['last_error'][name]
                d_term = kd * derivative
                self.state['last_error'][name] = error # Update for the next step

                # Combine the terms to get the final adjustment factor.
                adjustment = p_term + i_term + d_term
                # We use an exponent to ensure the adjustment factor is always positive,
                # preventing the loss weight from becoming negative.
                new_lambda = self.base_weights[name] * np.exp(adjustment)

                # As a final safety measure, clip the weight to a reasonable range.
                final_lambdas[name] = float(np.clip(new_lambda, 0.1, 10.0))

        return final_lambdas


# ==============================================================================
# 2. Q-LEARNING CONTROLLER (For dynamically adjusting the learning rate)
# ==============================================================================
# PURPOSE: Finding the best learning rate schedule is a classic deep learning
# problem. This controller frames it as a Reinforcement Learning problem. An "agent"
# learns a policy to adjust the learning rate to maximize a reward signal, where
# the reward is based on how quickly the training loss is decreasing.
#
# RL Formulation:
# - State: The recent average of the main training loss, discretized into buckets.
# - Actions: A small set of multiplicative changes to the LR (e.g., x0.9, x1.0, x1.1).
# - Reward: Calculated from the slope of the loss curve over a recent window.
#           A steep negative slope (fast improvement) gives a high reward.
#           A positive slope (worsening loss) gives a large penalty.

# --- Q-Learner Configuration ---
Q_CONTROLLER_CONFIG = {
    # RL agent parameters
    "q_table_size": 100,        # Granularity of state space: how many buckets to discretize the loss into.
    "num_lr_actions": 5,        # The number of possible actions the agent can take.
    "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1], # The actions themselves: multiply current LR by these factors.
    "learning_rate_q": 0.1,     # Alpha: How quickly the agent updates its Q-table (its knowledge).
    "discount_factor_q": 0.9,   # Gamma: How much the agent values future rewards over immediate ones.
    "exploration_rate_q": 0.3,  # Epsilon (initial): The initial chance of taking a random action to explore.
    "min_exploration_rate": 0.05,# Epsilon (minimum): The minimum exploration rate after decay.
    "exploration_decay": 0.9998, # The rate at which epsilon decays, shifting from exploration to exploitation.

    # LR and Loss bounds
    "lr_min": 5e-5,             # A hard floor for the learning rate.
    "lr_max": 5e-3,             # A hard ceiling for the learning rate.
    "loss_min": 0.001,          # The expected minimum loss, used for state bucket calculation.
    "loss_max": 1.0,            # The expected maximum loss, used for state bucket calculation.

    # Reward calculation parameters
    "metric_history_len": 500,  # How many recent loss values to store in the circular buffer.
    "trend_window": 100,        # The number of recent steps to use for calculating the loss slope.
    "improve_threshold": 1e-5,  # The loss slope must be steeper (more negative) than this to be "improving".
    "regress_threshold": 1e-6,  # The loss slope must be flatter (more positive) than this to be "regressing".
    "regress_penalty": -2.0,    # The reward (penalty) for an increasing loss.
    "stagnation_penalty": -0.5, # The reward (penalty) for a flat/stagnated loss.

    # Warmup parameters
    "warmup_steps": 500,        # Number of initial steps to use a fixed schedule instead of the agent.
    "warmup_lr_start": 1e-4,    # Starting LR for the warmup phase.
}

# --- Q-Learner State Representation ---
@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class QControllerState:
    """Stores all necessary state for the Q-Learner in a JAX-compatible format."""
    q_table: chex.Array          # The [state x action] table holding the agent's knowledge.
    metric_history: chex.Array   # A circular buffer of recent loss values.
    trend_history: chex.Array    # A circular buffer for calculating the loss slope.
    current_value: jnp.ndarray   # The current learning rate determined by the agent.
    exploration_rate: jnp.ndarray# The current value of epsilon.
    step_count: jnp.ndarray      # The global step counter for this controller.
    last_action_idx: jnp.ndarray # The index of the last action taken, needed for Q-table updates.
    last_reward: jnp.ndarray     # The last calculated reward, for logging.
    status_code: jnp.ndarray     # A code for the current status (0:Warmup, 1:Improving, etc.).

    def tree_flatten(self):
        children = (self.q_table, self.metric_history, self.trend_history,
                    self.current_value, self.exploration_rate, self.step_count,
                    self.last_action_idx, self.last_reward, self.status_code)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# --- Q-Learner Core Functions (designed to be JIT-compiled) ---

def init_q_controller(config: Dict[str, Any]) -> QControllerState:
    """Initializes the Q-Learner's state based on the configuration."""
    return QControllerState(
        q_table=jnp.zeros((config["q_table_size"], config["num_lr_actions"]), dtype=jnp.float32),
        metric_history=jnp.full((config["metric_history_len"],), (config["loss_min"] + config["loss_max"]) / 2, dtype=jnp.float32),
        trend_history=jnp.zeros((config["trend_window"],), dtype=jnp.float32),
        current_value=jnp.array(config["warmup_lr_start"], dtype=jnp.float32),
        exploration_rate=jnp.array(config["exploration_rate_q"], dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
        status_code=jnp.array(0, dtype=jnp.int32) # Start in warmup state
    )

@jax.jit
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey) -> QControllerState:
    """
    Decides the next learning rate. This is the "act" part of the RL loop.
    It follows a simple schedule during warmup, then uses the Q-table with an
    epsilon-greedy strategy to pick an action.
    """
    config = Q_CONTROLLER_CONFIG

    def warmup_action():
        # During warmup, simply follow a linear ramp-up of the learning rate.
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]
        new_value = config["warmup_lr_start"] * (1 - alpha) + config["lr_max"] * 0.5 * alpha
        return replace(state, current_value=new_value, step_count=state.step_count + 1, status_code=jnp.array(0))

    def regular_action():
        # 1. Determine current state: Convert the continuous average loss into a discrete state index.
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - 5, 5))
        state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)

        # 2. Epsilon-greedy action selection:
        # With probability epsilon, explore by choosing a random action.
        # Otherwise, exploit by choosing the best-known action from the Q-table for the current state.
        explore_key, action_key = jax.random.split(key)
        action_idx = jax.lax.cond(
            jax.random.uniform(explore_key) < state.exploration_rate,
            lambda: jax.random.randint(action_key, (), 0, config["num_lr_actions"]), # Explore
            lambda: jnp.argmax(state.q_table[state_idx])                             # Exploit
        )

        # 3. Apply action: Calculate the new learning rate by multiplying by the chosen factor.
        new_value = jnp.clip(state.current_value * jnp.array(config["lr_change_factors"])[action_idx], config["lr_min"], config["lr_max"])
        return replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)

    # Use JAX's conditional to switch between warmup and regular logic.
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)

@jax.jit
def q_controller_update(state: QControllerState, metric_value: float) -> QControllerState:
    """
    Updates the Q-table based on the reward from the last action. This is the "learn" part of the loop.
    It is called *after* a training step has been performed with the learning
    rate chosen by `q_controller_choose_action`.
    """
    config = Q_CONTROLLER_CONFIG
    metric_value_f32 = metric_value.astype(jnp.float32)

    # Update history buffers with the new loss value from the training step.
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value_f32)
    new_trend_history = jnp.roll(state.trend_history, -1).at[-1].set(metric_value_f32)

    def perform_update(st):
        # 1. Calculate Reward: Fit a line to the recent loss history to find the slope (trend).
        x = jnp.arange(config["trend_window"], dtype=jnp.float32)
        y = new_trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]

        # 2. Determine reward and status based on the calculated slope.
        status_code, reward = jax.lax.cond(
            slope < -config["improve_threshold"],
            lambda: (jnp.array(1), abs(slope) * 1000.0), # Improving -> Positive reward proportional to improvement.
            lambda: jax.lax.cond(
                slope > config["regress_threshold"],
                lambda: (jnp.array(3), config["regress_penalty"]),  # Regressing -> Large penalty.
                lambda: (jnp.array(2), config["stagnation_penalty"])# Stagnated -> Small penalty.
            )
        )

        # 3. Get the state index before the action (last_state_idx) and after (next_state_idx).
        old_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config["metric_history_len"]-5, 5))
        last_state_idx = jnp.clip(((old_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)

        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(new_metric_history, config["metric_history_len"]-5, 5))
        next_state_idx = jnp.clip(((new_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)

        # 4. Bellman Equation: Update the Q-table using the received reward.
        # This is the core learning step of the Q-learning algorithm.
        current_q = st.q_table[last_state_idx, st.last_action_idx]
        max_next_q = jnp.max(st.q_table[next_state_idx]) # Estimate of optimal future value.
        new_q = current_q + config["learning_rate_q"] * (reward + config["discount_factor_q"] * max_next_q - current_q)

        new_q_table = st.q_table.at[last_state_idx, st.last_action_idx].set(new_q.astype(st.q_table.dtype))

        # 5. Decay the exploration rate to favor exploitation as training progresses.
        new_exp_rate = jnp.maximum(config["min_exploration_rate"], st.exploration_rate * config["exploration_decay"])

        return replace(st, q_table=new_q_table, exploration_rate=new_exp_rate, last_reward=reward.astype(st.last_reward.dtype), status_code=status_code)

    # Only update the Q-table after warmup and once the trend window is full of real data.
    can_update = (state.step_count > config["warmup_steps"]) & \
                 (state.step_count > config["trend_window"]) & \
                 (state.last_action_idx >= 0)
    new_state = jax.lax.cond(can_update, perform_update, lambda s: s, state)

    # Always return the state with updated history buffers.
    return replace(new_state, metric_history=new_metric_history, trend_history=new_trend_history)


# ==============================================================================
# 3. DEMONSTRATION
# ==============================================================================
def main():
    """Run simulated demonstrations for both controllers."""

    print("\n" + "="*80)
    print("DEMONSTRATION 1: PID Controller for Loss Weights")
    print("="*80)
    print("Simulating a scenario with two losses, 'mae' and 'kld'.")
    print("The controller will try to adjust weights to steer the losses to their targets.\n")

    # Setup the controller with targets and tuning gains.
    pid_controller = PIDLambdaController(
        targets={'mae': 0.1, 'kld': 0.5},
        base_weights={'mae': 1.0, 'kld': 0.2},
        gains={'mae': (0.5, 0.01, 0.1), 'kld': (0.8, 0.02, 0.2)}
    )

    # Simulate a training process where loss values are noisy but trend towards their targets.
    current_metrics = {'mae': 0.3, 'kld': 0.2}
    for i in range(20):
        # The controller calculates new weights based on the last step's metrics.
        new_weights = pid_controller(current_metrics)

        print(
            f"Step {i+1:2d}: "
            f"MAE Loss = {current_metrics['mae']:.3f} (Target: {pid_controller.targets['mae']:.3f}) -> New Weight = {new_weights['mae']:.3f} | "
            f"KLD Loss = {current_metrics['kld']:.3f} (Target: {pid_controller.targets['kld']:.3f}) -> New Weight = {new_weights['kld']:.3f}"
        )

        # Simulate the effect of the new weights on the next step's loss values.
        # The losses slowly move towards their targets, with some random noise.
        current_metrics['mae'] += (pid_controller.targets['mae'] - current_metrics['mae']) * 0.2 + np.random.randn() * 0.02
        current_metrics['kld'] += (pid_controller.targets['kld'] - current_metrics['kld']) * 0.2 + np.random.randn() * 0.05

    print("\n\n" + "="*80)
    print("DEMONSTRATION 2: Q-Learning Controller for Learning Rate")
    print("="*80)
    print("Simulating a training loop where the loss depends on the chosen LR.")
    print("The agent will learn to pick an LR close to the 'optimal' value of 1e-3.\n")

    # Initialize the agent's state and other simulation variables.
    q_state = init_q_controller(Q_CONTROLLER_CONFIG)
    key = jax.random.PRNGKey(42)
    optimal_lr = 1e-3 # The "ground truth" best LR in this simulation.
    q_status_map = {0: "Warmup", 1: "Improving", 2: "Stagnated", 3: "Regressing"}

    # Push the initial state to the accelerator (GPU/TPU) for JIT compilation.
    q_state_device = jax.device_put(q_state)

    for step in range(2000):
        # === The core RL loop ===
        # 1. Choose an action (get a new learning rate from the agent).
        key, action_key = jax.random.split(key)
        q_state_device = q_controller_choose_action(q_state_device, action_key)
        current_lr = q_state_device.current_value

        # 2. Simulate a training step and get a resulting loss.
        # This simple formula creates a "loss landscape" where the minimum loss
        # occurs when the agent's `current_lr` is equal to the `optimal_lr`.
        simulated_loss = (np.log10(current_lr) - np.log10(optimal_lr))**2 + np.random.uniform(0.01, 0.02)
        simulated_loss = np.clip(simulated_loss, Q_CONTROLLER_CONFIG['loss_min'], Q_CONTROLLER_CONFIG['loss_max'])

        # 3. Update the Q-Learner: The agent learns from the outcome (the loss).
        q_state_device = q_controller_update(q_state_device, simulated_loss)

        # Print progress periodically.
        if step % 100 == 0 or step == Q_CONTROLLER_CONFIG['warmup_steps']:
            # Copy state back from device to host (CPU) for printing.
            q_state_host = jax.device_get(q_state_device)
            status = q_status_map[int(q_state_host.status_code)]
            print(
                f"Step {step:4d}: "
                f"LR = {float(q_state_host.current_value):.2e}, "
                f"Loss = {simulated_loss:.4f}, "
                f"Reward = {float(q_state_host.last_reward):+6.2f}, "
                f"Explore Rate = {float(q_state_host.exploration_rate):.2f}, "
                f"Status = {status}"
            )
        time.sleep(0.001) # Slow down simulation for readability.


if __name__ == "__main__":
    main()