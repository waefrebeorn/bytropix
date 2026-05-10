# Wubu_Clockwork_Perceptron.py
#
# THE TRI-CAMERAL MIND: VECTOR SPACE EDITION
#
# Scenario: "The Drifting Horizon"
# A 2D classification boundary rotates 180 degrees.
# The AI must adapt its weights (Body) while storing the
# rotational momentum in the Beach (Soul).
#
# This proves Geodesic AI works on Matrices, not just Scalars.

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import chex
import numpy as np
from typing import NamedTuple, Dict, Tuple, Any

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. GEODESIC HARDWARE (Vector-Ready)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float) -> optax.GradientTransformation:
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
        
        # Symmetric Decomposition (Works on any shape matrix)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), updates, quotients)
        
        # Storage (The Soul Matrix)
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        # Body Update (The Weight Matrix)
        new_moment1 = optax.incremental_update(remainders, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        # Standardized Update (Scaling handled externally by Q-Learner)
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
# 2. THE EXECUTIVES (The Ghost in the Machine)
# ==============================================================================

class PIDSensitivity:
    """Regulates how much the Soul influences the Body."""
    def __init__(self):
        self.kp, self.ki, self.kd = 1.5, 0.2, 0.1
        self.integral = 0.0
        self.last_error = 0.0
        self.val = 0.01

    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -2.0, 2.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        self.val = np.clip(self.val + adj * 0.05, 0.001, 0.5) # Max 50% influence
        return self.val

class QLearnerLR:
    """Regulates how malleable the Body is."""
    def __init__(self):
        self.lr = 0.05
        self.table = np.zeros((10, 3)) # States based on Loss magnitude
        self.state = 0
        self.action = 1
    
    def step(self, loss):
        # Reward is stability (low loss)
        reward = 1.0 / (loss + 1e-5)
        next_state = min(int(loss * 20), 9)
        
        # Update Q-Table (Simplified)
        target = reward + 0.9 * np.max(self.table[next_state])
        self.table[self.state, self.action] += 0.1 * (target - self.table[self.state, self.action])
        
        # Epsilon-greedy Action
        if np.random.rand() < 0.1: action = np.random.randint(0, 3)
        else: action = np.argmax(self.table[next_state])
        
        # 0=Decrease, 1=Hold, 2=Increase
        if action == 0: self.lr *= 0.8
        elif action == 2: self.lr *= 1.2
        self.lr = np.clip(self.lr, 1e-5, 0.2)
        
        self.state = next_state
        self.action = action
        return self.lr

# ==============================================================================
# 3. THE CLOCKWORK EXPERIMENT
# ==============================================================================

def geodesic_neuron_forward(params, opt_state, x_input, sensitivity):
    # W is a [2, 1] matrix (Input 2D -> Output 1D)
    W_body = params['W']
    
    # The Soul is also a [2, 1] matrix stored in the Beach
    soul_W = opt_state.stored_topology['W']
    echo_W = opt_state.stored_residue['W']
    W_soul = (soul_W * (2 * jnp.pi)) + echo_W
    
    # Effective Weight = Body - (Sensitivity * Soul)
    # If the Soul records a massive "Right Drift", this term 
    # subtracts that drift from the Body's impulse.
    W_effective = W_body - (sensitivity * W_soul)
    
    return jnp.dot(x_input, W_effective)

def run_clockwork_perceptron():
    print("\n" + "="*80)
    print("THE CLOCKWORK PERCEPTRON: 2D VECTOR SPACE TEST")
    print("Task: Track a rotating decision boundary (0 to 180 degrees).")
    print("="*80)

    # --- SETUP ---
    TOTAL_STEPS = 100
    
    # Initialize Weights (2 inputs -> 1 output)
    # We start pointing UP (0, 1)
    params = {'W': jnp.array([[0.0], [1.0]], dtype=jnp.float64)}
    
    opt = geodesic_optimizer(learning_rate=1.0)
    state = opt.init(params)
    
    pid = PIDSensitivity()
    q_agent = QLearnerLR()
    
    print(f"{'STEP':<4} | {'ANGLE':<5} | {'BODY VECTOR':<20} | {'SOUL VECTOR':<20} | {'LOSS':<6} | {'STATUS'}")
    print("-" * 90)

    for t in range(TOTAL_STEPS):
        # 1. GENERATE DATA (The Moving Truth)
        # The Angle rotates from 90 deg (Up) to 270 deg (Down)
        angle_rad = jnp.pi/2 + (t / TOTAL_STEPS) * jnp.pi 
        
        # The "True" vector points in this direction
        true_vector = jnp.array([jnp.cos(angle_rad), jnp.sin(angle_rad)])
        
        # Input is a random point on the unit circle
        key = jax.random.PRNGKey(t)
        random_angle = jax.random.uniform(key, minval=0, maxval=2*jnp.pi)
        x_input = jnp.array([jnp.cos(random_angle), jnp.sin(random_angle)])
        
        # Ground Truth: Dot product with the True Vector
        # (Positive if aligned, Negative if opposed)
        y_target = jnp.dot(x_input, true_vector)

        # 2. EXECUTIVES
        sens = pid.val
        lr = q_agent.lr
        
        # 3. INFERENCE
        # Note: reshaping x_input for matrix math
        x_in_mat = x_input.reshape(1, 2) 
        y_pred = geodesic_neuron_forward(params, state, x_in_mat, sensitivity=sens)[0,0]
        
        loss = (y_pred - y_target)**2
        
        # 4. UPDATE MANAGERS
        pid.update(float(loss))
        lr = q_agent.step(float(loss))
        
        # 5. UPDATE HARDWARE
        # Gradient of MSE w.r.t W is roughly: 2 * (pred - target) * x
        # We approximate the force as simply the error direction vector
        error = y_pred - y_target
        grad_W = error * x_input.reshape(2, 1)
        
        # Stress Factor (Pain Amplification from previous tuning)
        grad_W *= 2.0 
        
        grads = {'W': grad_W}
        updates, state = opt.update(grads, state, params)
        
        # Apply Descent (-LR)
        scaled_updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
        params = optax.apply_updates(params, scaled_updates)

        # 6. LOGGING
        if t % 10 == 0 or t == TOTAL_STEPS - 1:
            # Formatting vector strings for display
            w_body = params['W'].flatten()
            body_str = f"[{w_body[0]:.2f}, {w_body[1]:.2f}]"
            
            s_idx = state.stored_topology['W'].flatten()
            soul_str = f"[{s_idx[0]:.0f}, {s_idx[1]:.0f}]" # Just printing integer winds
            
            # Check if we are tracking
            # Dot product of Body and True Vector should be close to 1.0
            alignment = jnp.dot(w_body, true_vector)
            status = "✅" if alignment > 0.9 else "⚠️"
            if alignment < 0.5: status = "❌"
            
            degrees = int(jnp.degrees(angle_rad))
            print(f"{t:<4} | {degrees:<5} | {body_str:<20} | {soul_str:<20} | {loss:<6.4f} | {status}")

    print("-" * 90)
    print("ANALYSIS:")
    print("The 'Body Vector' should rotate to match the Angle.")
    print("The 'Soul Vector' (Beach) should show the accumulated rotational drag.")

if __name__ == "__main__":
    run_clockwork_perceptron()