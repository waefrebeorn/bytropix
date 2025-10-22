import os
# --- THE EFFICIENCY FIX ---
# This script is locked to the CPU to ensure it runs efficiently without GPU overhead.
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
# This is the final, definitive version. The agent's design is complete. It now
# faces the Ultimate Graduation Exam.
#
# THE ULTIMATE TEST: The "Shifting Sands"
# The loss landscape is no longer static. The optimal learning rate is now a
# moving target that changes over time, simulating a real training run. The agent
# must not only find the optimum, but adapt and track it. This is the final
# test of a truly intelligent, adaptive system.
# ==============================================================================


# ==============================================================================
# 1. THE SELF-AWARE Q-LEARNING CONTROLLER - FINAL, LOCKED DESIGN
# ==============================================================================
Q_CONTROLLER_CONFIG = {
    "q_table_size": 1000,
    "num_lr_actions": 4,
    "lr_change_factors": [0.5, 0.9, 1.1, 3.0],
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
    print("\n" + "="*80 + "\nDEMONSTRATION: The Ultimate Graduation Exam - The Shifting Sands (Marathon)\n" + "="*80)
    print("The optimal LR is now a moving target over 10,000 steps.")
    print("Success requires the agent to not just find the goal, but to adapt and track it.\n")

    meta_controller = MetaController()
    current_loss = 0.5
    total_steps = 10000
    
    # --- THE SHIFTING SANDS LANDSCAPE ---
    def get_optimal_lr_for_step(step):
        """The optimal LR changes over the course of the simulation."""
        # Phase 1: High LR for initial learning
        if step < 3000:
            return 2e-3
        # Phase 2: Smoothly decay to a medium LR for refinement
        elif step < 6000:
            return np.interp(step, [3000, 6000], [2e-3, 8e-4])
        # Phase 3: Smoothly decay to a tiny LR for fine-tuning
        else:
            return np.interp(step, [6000, total_steps], [8e-4, 1e-4])

    def get_loss_from_landscape(lr, optimal_lr):
        distance_from_optimum_sq = (np.log10(lr) - np.log10(optimal_lr))**2
        base_loss = np.tanh(10 * distance_from_optimum_sq)
        noise = np.random.uniform(-0.01, 0.01)
        return np.clip(base_loss + noise, 0.001, 1.0)

    # --- THE FINAL EXAM LOOP ---
    for step in range(total_steps):
        lr_for_this_step = float(meta_controller.q_state.current_value)
        
        # The ground truth is now dynamic
        optimal_lr = get_optimal_lr_for_step(step)
        new_loss = get_loss_from_landscape(lr_for_this_step, optimal_lr)
        
        meta_controller.step(prev_loss=current_loss, new_loss=new_loss)
        
        if step % 50 == 0 or step == Q_CONTROLLER_CONFIG['warmup_steps'] or step == total_steps - 1:
            reward = float(meta_controller.q_state.last_reward)
            exploration_rate = float(meta_controller.q_state.exploration_rate)
            print(f"Step {step:5d}: Agent LR={lr_for_this_step:.2e}, Optimal LR={optimal_lr:.2e}, "
                  f"Loss={new_loss:.3f}, Rwd={reward:+6.2f}, Explore={exploration_rate:.2f}")
        
        current_loss = new_loss

if __name__ == "__main__":
    main()