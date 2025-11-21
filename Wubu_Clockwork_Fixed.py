# Wubu_Clockwork_Fixed.py
#
# THE TRI-CAMERAL MIND: CLOCKWORK EDITION [WITH TRANSMISSION]
#
# Change: Implemented "Gear Ratio" (Mechanical Advantage).
# Principle: 
#   - Inputs to Soul are amplified (x100) to trigger winding.
#   - Inputs to Body are dampened (/100) to maintain stability.
#   - This allows the Soul to track "Micro-Drifts" as "Macro-Events".

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import chex
import numpy as np
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE GEODESIC ENGINE (With Gear Ratio)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float, gear_ratio: float = 100.0) -> optax.GradientTransformation:
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
        
        # --- THE TRANSMISSION ---
        # 1. Amplify the signal to wake up the Soul
        amplified_updates = jax.tree_util.tree_map(lambda g: g * gear_ratio, updates)
        
        # 2. Decompose the AMPLIFIED signal
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified_updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified_updates, quotients)
        
        # 3. Store the AMPLIFIED history (The Soul remembers the high-energy intent)
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        # 4. Dampen the signal for the Body (Stability)
        # We scale the remainder back down so the weights don't explode.
        # The Body only sees the "residue" of the massive winding event.
        body_updates = jax.tree_util.tree_map(lambda r: r / gear_ratio, remainders)

        # 5. Update Weights
        new_moment1 = optax.incremental_update(body_updates, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        final_updates = jax.tree_util.tree_map(
            lambda m1, m2: -learning_rate * m1 / (jnp.sqrt(m2) + 1e-8), 
            m1_hat, m2_hat
        )
        
        return final_updates, GeodesicState(
            count=count, moment1=new_moment1, moment2=new_moment2, 
            stored_topology=new_topology, stored_residue=new_residue
        )
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# 2. THE EXECUTIVES
# ==============================================================================

class PIDSensitivity:
    def __init__(self):
        self.kp, self.ki, self.kd = 2.0, 0.5, 0.1
        self.integral = 0.0
        self.last_error = 0.0
        self.val = 0.01

    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -5.0, 5.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        # Allow higher sensitivity range
        self.val = np.clip(self.val + adj * 0.1, 0.0, 1.0)
        return self.val

class QLearnerLR:
    def __init__(self):
        self.lr = 0.1 # Start more aggressive
        self.table = np.zeros((10, 3)) 
        self.state = 0
        self.action = 1
    
    def step(self, loss):
        reward = 1.0 / (loss + 1e-5)
        next_state = min(int(loss * 20), 9)
        target = reward + 0.9 * np.max(self.table[next_state])
        self.table[self.state, self.action] += 0.1 * (target - self.table[self.state, self.action])
        
        if np.random.rand() < 0.15: action = np.random.randint(0, 3) # More explore
        else: action = np.argmax(self.table[next_state])
        
        if action == 0: self.lr *= 0.8
        elif action == 2: self.lr *= 1.2
        self.lr = np.clip(self.lr, 0.001, 0.5) # Prevent freezing (Min LR 0.001)
        
        self.state = next_state
        self.action = action
        return self.lr

# ==============================================================================
# 3. THE CLOCKWORK EXPERIMENT (GEARED)
# ==============================================================================

def geodesic_neuron_forward(params, opt_state, x_input, sensitivity, gear_ratio):
    W_body = params['W']
    
    soul_W = opt_state.stored_topology['W']
    echo_W = opt_state.stored_residue['W']
    
    # Reconstruct History (Account for Gear Ratio!)
    # Total = (Soul * 2pi + Echo) / Gear_Ratio
    # This scales the massive soul history back down to "Reality Scale"
    W_history = ((soul_W * (2 * jnp.pi)) + echo_W) / gear_ratio
    
    # In this Rotation task, the History represents the "Inertia" of the rotation.
    # If we want to TRACK the rotation, we should ADD the inertia (Momentum).
    # If we wanted to STOP the rotation (Tired Runner), we would SUBTRACT.
    # Here, we use the Soul as a Flywheel to help us turn.
    
    # BUT, let's stick to the PID logic: The PID tries to correct error.
    # If the body is lagging, the error is high.
    # We want the Soul to push the Body towards the Target.
    
    W_effective = W_body + (sensitivity * W_history) 
    
    return jnp.dot(x_input, W_effective)

def run_clockwork_geared():
    print("\n" + "="*80)
    print("THE CLOCKWORK PERCEPTRON: GEARED TRANSMISSION")
    print("Gear Ratio: 100.0 (High Sensitivity to Micro-Drift)")
    print("Strategy: Soul acts as Momentum (Flywheel), helping the turn.")
    print("="*80)

    TOTAL_STEPS = 100
    params = {'W': jnp.array([[0.0], [1.0]], dtype=jnp.float64)} # Start UP
    
    GEAR_RATIO = 100.0
    opt = geodesic_optimizer(learning_rate=1.0, gear_ratio=GEAR_RATIO)
    state = opt.init(params)
    
    pid = PIDSensitivity()
    q_agent = QLearnerLR()
    
    print(f"{'STEP':<4} | {'ANGLE':<5} | {'BODY VECTOR':<16} | {'SOUL (Winds)':<16} | {'LOSS':<6} | {'STATUS'}")
    print("-" * 80)

    for t in range(TOTAL_STEPS):
        # Angle: 90 -> 270 (Rotating Left/CCW in math terms, or just Downward)
        angle_rad = jnp.pi/2 + (t / TOTAL_STEPS) * jnp.pi 
        true_vector = jnp.array([jnp.cos(angle_rad), jnp.sin(angle_rad)])
        
        # Random Input
        key = jax.random.PRNGKey(t)
        ang = jax.random.uniform(key, minval=0, maxval=2*jnp.pi)
        x_input = jnp.array([jnp.cos(ang), jnp.sin(ang)])
        y_target = jnp.dot(x_input, true_vector)

        # Executives
        sens = pid.val
        lr = q_agent.lr
        
        # Inference
        x_in_mat = x_input.reshape(1, 2) 
        # Note: Using +sensitivity (Momentum Mode)
        y_pred = geodesic_neuron_forward(params, state, x_in_mat, sens, GEAR_RATIO)[0,0]
        
        loss = (y_pred - y_target)**2
        
        # Updates
        pid.update(float(loss))
        lr = q_agent.step(float(loss))
        
        # Hardware Update
        error = y_pred - y_target
        grad_W = error * x_input.reshape(2, 1)
        
        # Standard update (The Gear Ratio inside optimizer handles the amplification)
        grads = {'W': grad_W}
        updates, state = opt.update(grads, state, params)
        
        scaled_updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
        params = optax.apply_updates(params, scaled_updates)

        # Logging
        if t % 10 == 0 or t == TOTAL_STEPS - 1:
            w = params['W'].flatten()
            body_str = f"[{w[0]:.2f}, {w[1]:.2f}]"
            
            s = state.stored_topology['W'].flatten()
            soul_str = f"[{s[0]:.0f}, {s[1]:.0f}]"
            
            # Alignment check
            align = jnp.dot(w, true_vector)
            status = "✅" if align > 0.9 else "⚠️"
            if align < 0.5: status = "❌"
            
            deg = int(jnp.degrees(angle_rad))
            print(f"{t:<4} | {deg:<5} | {body_str:<16} | {soul_str:<16} | {loss:<6.4f} | {status}")

    print("-" * 80)
    print("Final Analysis:")
    w_final = params['W'].flatten()
    print(f"Final Body: {w_final}")
    print(f"True Down : [0.0, -1.0] (approx)")
    print(f"Soul State: {state.stored_topology['W'].flatten()}")
    
    if w_final[1] < -0.8:
        print(">>> SUCCESS: The Body successfully rotated 180 degrees.")
        print(">>> The Soul accumulated winding numbers to track the rotation.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_clockwork_geared()