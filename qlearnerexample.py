import os
# --- THE EFFICIENCY FIX ---
# This is the most important line for performance. It tells JAX to ONLY use the CPU.
# For a simple simulation like this, using the GPU creates massive, unnecessary overhead.
# This makes the script lightweight and fast, as it should be.
os.environ['JAX_PLATFORMS'] = 'cpu'

import time
import numpy as np
import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, Tuple, Any

# JAX can be quite verbose with warnings; this helps keep the demo output clean.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# SCRIPT OVERVIEW: THE FINAL "SELF-AWARE AGENT"
# ==============================================================================
# This is the final, definitive version. The agent's design is complete.
# It now faces the Superhard Final Exam.
#
# THE SUPERHARD TEST: The "Poisoned Chalice"
# The simulation is now longer (20,000 steps) and harder. The landscape
# contains a deep, attractive local minimum (the "poisoned chalice") and an
# extremely narrow global minimum. The noise is higher. Success requires a
# persistent, intelligent search and proves the robustness of the final design.
# Data is printed every 25 steps for high-resolution analysis.
# ==============================================================================


# ==============================================================================
# 1. THE SELF-AWARE Q-LEARNING CONTROLLER - FINAL, LOCKED DESIGN
# ==============================================================================
Q_CONTROLLER_CONFIG = {
    "q_table_size": 1000,
    "num_lr_actions": 4,
    "lr_change_factors": [0.5, 0.9, 1.1, 3.0],
    # The agent is now more "farsighted" to encourage seeking the global optimum.
    "learning_rate_q": 0.01, "discount_factor_q": 0.95,
    "lr_min": 5e-5, "lr_max": 5e-3,
    "loss_min": 0.001, "loss_max": 1.0,
    "min_exploration_rate": 0.1,
    "max_exploration_rate": 1.0,
    "warmup_steps": 500, "warmup_lr_start": 1e-4,
}

# --- Q-Learner State and Functions ---
@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class QControllerState:
    q_table: chex.Array
    current_value: jnp.ndarray; exploration_rate: jnp.ndarray; step_count: jnp.ndarray
    last_action_idx: jnp.ndarray; last_reward: jnp.ndarray
    def tree_flatten(self):
        return (self.q_table, self.current_value, self.exploration_rate,
                self.step_count, self.last_action_idx, self.last_reward), None
    @classmethod
    def tree_unflatten(cls, aux_data, children): return cls(*children)

def init_q_controller(config: Dict[str, Any]) -> QControllerState:
    q_table = jnp.zeros((config["q_table_size"], config["num_lr_actions"]), dtype=jnp.float32)
    increase_action_indices = jnp.array([2, 3])
    optimistic_bias = 0.1
    q_table = q_table.at[:, increase_action_indices].set(optimistic_bias)
    return QControllerState(
        q_table=q_table,
        current_value=jnp.array(config["warmup_lr_start"], dtype=jnp.float32),
        exploration_rate=jnp.array(config["max_exploration_rate"], dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32))

@jax.jit
def q_controller_step(state: QControllerState, key: chex.PRNGKey, prev_loss: float, new_loss: float) -> QControllerState:
    """A unified function that chooses an action AND learns from the result."""
    config = Q_CONTROLLER_CONFIG
    
    # --- Action Selection (based on prev_loss) ---
    state_idx = jnp.clip(((prev_loss - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
    explore_key, action_key = jax.random.split(key)
    action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate,
        lambda: jax.random.randint(action_key, (), 0, config["num_lr_actions"]),
        lambda: jnp.argmax(state.q_table[state_idx]))
    new_value = jnp.clip(state.current_value * jnp.array(config["lr_change_factors"])[action_idx], config["lr_min"], config["lr_max"])
    state_after_action = replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)

    # --- Learning (based on the transition from prev_loss to new_loss) ---
    ## BIGGER LAMBDA: The reward for low loss is 10x stronger.
    reward = (1.0 / (new_loss + 1e-6)) * 1.0
    improvement_bonus = jax.lax.cond(new_loss < prev_loss, lambda: 0.1, lambda: -0.1)
    total_reward = reward + improvement_bonus
    last_state_idx = state_idx
    next_state_idx = jnp.clip(((new_loss - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
    current_q = state_after_action.q_table[last_state_idx, state_after_action.last_action_idx]
    max_next_q = jnp.max(state_after_action.q_table[next_state_idx])
    new_q = current_q + config["learning_rate_q"] * (total_reward + config["discount_factor_q"] * max_next_q - current_q)
    new_q_table = state_after_action.q_table.at[last_state_idx, state_after_action.last_action_idx].set(new_q.astype(state_after_action.q_table.dtype))
    
    return replace(state_after_action, q_table=new_q_table, last_reward=total_reward.astype(state_after_action.last_reward.dtype))

# ==============================================================================
# 2. META-CONTROLLER and DEMONSTRATION
# ==============================================================================
class MetaController:
    """Manages the Self-Aware Q-Learner."""
    def __init__(self):
        self.q_config = Q_CONTROLLER_CONFIG
        self.q_state = init_q_controller(self.q_config)
        self.key = jax.random.PRNGKey(42)

    def step(self, prev_loss: float, new_loss: float):
        """Performs one full, synchronized step of the control loop."""
        
        # --- THE SELF-AWARENESS MECHANISM ---
        exploration_command = np.interp(
            prev_loss,
            [0.0, 1.0], # Input range (loss)
            [self.q_config["min_exploration_rate"], self.q_config["max_exploration_rate"]] # Output range (exploration)
        )
        self.q_state = replace(self.q_state, exploration_rate=jnp.array(exploration_command))

        # The agent acts and learns in a single, unified, JIT-compiled step.
        self.key, step_key = jax.random.split(self.key)
        self.q_state = q_controller_step(self.q_state, step_key, prev_loss, new_loss)

def main():
    print("\n" + "="*80 + "\nDEMONSTRATION: The Marathon Final Exam - The Poisoned Chalice\n" + "="*80)
    print("The simulation is now 20,000 steps with a much harder loss landscape.")
    print("This is the ultimate test of the complete system.\n")

    meta_controller = MetaController()
    current_loss = 0.5
    
    # --- THE "POISONED CHALICE" LANDSCAPE ---
    log_lr_global_min = np.log10(1e-3)  # The extremely narrow deep valley
    log_lr_local_min = np.log10(5e-5)   # The deep, attractive pothole trap

    def get_loss_from_landscape(lr):
        log_lr = np.log10(lr)
        # Global Minimum: Extremely deep (1.0) but also extremely narrow (0.05).
        valley = 1.0 * np.exp(-((log_lr - log_lr_global_min)**2) / (2 * 0.05**2))
        # Local Minimum: Very deep (0.9), making it a very tempting trap.
        pothole = 0.9 * np.exp(-((log_lr - log_lr_local_min)**2) / (2 * 0.3**2))
        base_loss = 1.0 - valley - pothole
        # Higher noise to obscure the signals.
        noise = np.random.uniform(-0.05, 0.05)
        return np.clip(base_loss + noise, 0.001, 1.0)

    # --- LONGER SIMULATION & HIGH-FREQUENCY LOGGING ---
    total_steps = 20000
    for step in range(total_steps):
        lr_for_this_step = float(meta_controller.q_state.current_value)
        
        # After warmup, start the agent deep in the trap (the local minimum).
        if step == Q_CONTROLLER_CONFIG['warmup_steps']:
            lr_for_this_step = 5e-5

        new_loss = get_loss_from_landscape(lr_for_this_step)
        
        meta_controller.step(prev_loss=current_loss, new_loss=new_loss)
        
        # Print data every 25 steps for high-resolution analysis.
        if step % 25 == 0 or step == Q_CONTROLLER_CONFIG['warmup_steps'] or step == total_steps - 1:
            reward = float(meta_controller.q_state.last_reward)
            exploration_rate = float(meta_controller.q_state.exploration_rate)
            print(f"Step {step:5d}: LR={lr_for_this_step:.2e}, Loss={new_loss:.3f}, Rwd={reward:+7.2f} "
                  f"| Self-Aware Explore={exploration_rate:.2f}")
        
        current_loss = new_loss

if __name__ == "__main__":
    main()