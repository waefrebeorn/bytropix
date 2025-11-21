# Wubu_TriCameral_Mind.py
#
# THE TRI-CAMERAL ARCHITECTURE
#
# 1. HARDWARE: Geodesic Engine (Symmetric Storage)
# 2. EXECUTIVE 1: PID Controller (Regulates Soul Sensitivity/Alpha)
# 3. EXECUTIVE 2: Q-Learner (Regulates Body Plasticity/LR)
#
# Goal: Solve "The Tired Runner" with ZERO manual tuning.

import os
os.environ['JAX_PLATFORMS'] = 'cpu' # Efficiency fix

import jax
import jax.numpy as jnp
import optax
import chex
import numpy as np
from typing import NamedTuple, Dict, Tuple, Any
from dataclasses import dataclass, replace

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# PART 1: THE GEODESIC HARDWARE (The Storage Layer)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float) -> optax.GradientTransformation:
    # Note: LR is now dynamic, passed in update_fn if we wanted, 
    # but for Optax structure we scale the updates externally.
    def init_fn(params: optax.Params) -> GeodesicState:
        return GeodesicState(
            count=jnp.zeros([], jnp.int32),
            moment1=jax.tree_util.tree_map(jnp.zeros_like, params),
            moment2=jax.tree_util.tree_map(jnp.zeros_like, params),
            stored_topology=jax.tree_util.tree_map(jnp.zeros_like, params),
            stored_residue=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates: optax.Updates, state: GeodesicState, params=None) -> tuple[optax.Updates, GeodesicState]:
        boundary = 2 * jnp.pi
        
        # Symmetric Decomposition
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), updates, quotients)
        
        # Storage
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        # Descent Update
        new_moment1 = optax.incremental_update(remainders, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        # Standardizing update shape (Scaling happens in the loop via Q-Learner)
        final_updates = jax.tree_util.tree_map(
            lambda m1, m2: m1 / (jnp.sqrt(m2) + 1e-8), 
            m1_hat, m2_hat
        )
        return final_updates, GeodesicState(
            count=count, moment1=new_moment1, moment2=new_moment2, 
            stored_topology=new_topology, stored_residue=new_residue
        )
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# PART 2: THE EXECUTIVE LAYERS (Your Controllers)
# ==============================================================================

# --- A. THE PID AMYGDALA (Regulates Sensitivity) ---
class PIDSensitivityController:
    def __init__(self, target_error=0.0, kp=2.0, ki=0.5, kd=0.1):
        self.target = target_error
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.last_error = 0.0
        self.base_sensitivity = 0.01

    def update(self, current_error):
        error = abs(current_error) - self.target # We want error to be 0
        
        p_term = self.kp * error
        self.integral = np.clip(self.integral + error, -5.0, 5.0)
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.last_error)
        self.last_error = error
        
        # Activation function: Higher error -> Higher sensitivity
        adjustment = p_term + i_term + d_term
        # Sensitivity can range from 0.001 (ignore memory) to 1.0 (full memory override)
        new_sensitivity = float(np.clip(self.base_sensitivity + adjustment, 0.001, 1.0))
        return new_sensitivity

# --- B. THE Q-LEARNER CORTEX (Regulates Plasticity/LR) ---
# (Simplified version of your script for embedding)
class QLearnerLite:
    def __init__(self):
        self.q_table = np.zeros((10, 3)) # 10 states, 3 actions (Decrease, Hold, Increase)
        self.lr_options = [0.5, 1.0, 1.5] # Multipliers
        self.current_lr = 0.01
        self.state_idx = 0
        self.last_action = 1
        self.epsilon = 0.2

    def step(self, loss):
        # 1. Learn from previous
        reward = 1.0 / (loss + 1e-6)
        # Simple Q-update
        next_state_idx = min(int(loss * 10), 9)
        best_next = np.max(self.q_table[next_state_idx])
        self.q_table[self.state_idx, self.last_action] += 0.1 * (reward + 0.9 * best_next - self.q_table[self.state_idx, self.last_action])
        
        # 2. Act
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.q_table[next_state_idx])
            
        multiplier = self.lr_options[action]
        self.current_lr = np.clip(self.current_lr * multiplier, 1e-5, 0.1)
        
        self.state_idx = next_state_idx
        self.last_action = action
        return self.current_lr

# ==============================================================================
# PART 3: THE TRI-CAMERAL EXPERIMENT (Tired Runner Auto-Tuned)
# ==============================================================================

def geodesic_forward(params, opt_state, x, sensitivity):
    w = params['w']
    body = w * x
    
    soul = opt_state.stored_topology['w']
    echo = opt_state.stored_residue['w']
    history = (soul * (2 * jnp.pi)) + echo
    
    # The PID controller controls 'sensitivity'
    return body - (sensitivity * history)

def run_tricameral_mind():
    print("\n" + "="*80)
    print("THE TRI-CAMERAL MIND: AUTO-TUNED GEODESIC AI")
    print("Task: The Tired Runner (Decay 1.0 -> 0.0)")
    print("Executives: PID (Sensitivity) + Q-Learner (Learning Rate)")
    print("="*80)

    STEPS = 25
    inputs = jnp.ones(STEPS)
    targets = jnp.linspace(1.0, 0.0, STEPS)

    # 1. Hardware
    params = {'w': jnp.array(1.0)}
    opt = geodesic_optimizer(learning_rate=1.0) # Base LR, scaled by Q-Learner
    state = opt.init(params)

    # 2. Executives
    amygdala = PIDSensitivityController() # Manages the Soul
    cortex = QLearnerLite()               # Manages the Body

    current_sensitivity = 0.01
    current_lr = 0.01

    print(f"{'STEP':<4} | {'TGT':<6} | {'PRED':<6} | {'SENS (PID)':<10} | {'LR (Q)':<10} | {'SOUL'}")
    print("-" * 70)

    for i in range(STEPS):
        x = inputs[i]
        y_target = targets[i]
        
        # --- A. EXECUTIVE CONTROL (PID) ---
        # The Amygdala looks at the last error (or current state) and adjusts sensitivity.
        # If we are failing, Sensitivity goes UP -> Soul takes over -> Braking increases.
        # We do this *before* the forward pass prediction to simulate reactive control,
        # or *after* (next step) for predictive. Let's use current reactive.
        
        # 1. Inference (using current sensitivity)
        y_pred = geodesic_forward(params, state, x, sensitivity=current_sensitivity)
        
        error = y_pred - y_target
        
        # 2. Amygdala Update (For next step)
        current_sensitivity = amygdala.update(float(error))
        
        # 3. Cortex Update (Decide Learning Rate based on error/loss)
        loss = float(abs(error))
        current_lr = cortex.step(loss)

        # 4. Geodesic Update (Hardware)
        # We use the error as the gradient signal
        grads = {'w': jnp.array(error)}
        updates, state = opt.update(grads, state, params)
        
        # Apply updates with the Q-Learner's chosen Learning Rate
        # Note: We apply negative LR for descent
        scaled_updates = jax.tree_util.tree_map(lambda u: -current_lr * u, updates)
        params = optax.apply_updates(params, scaled_updates)
        
        # 5. Monitoring
        hist = (state.stored_topology['w'] * 2 * jnp.pi) + state.stored_residue['w']
        
        print(f"{i:<4} | {y_target:<6.2f} | {y_pred:<6.2f} | {current_sensitivity:<10.4f} | {current_lr:<10.6f} | {hist:<6.2f}")

    final_error = abs(y_pred - y_target)
    print("-" * 70)
    print(f"Final Prediction: {y_pred:.4f} (Target: {y_target:.4f})")
    
    if final_error < 0.05:
        print(">>> SUCCESS: The Tri-Cameral Mind solved the problem without manual tuning.")
        print(">>> The PID ramped up sensitivity as errors increased.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_tricameral_mind()