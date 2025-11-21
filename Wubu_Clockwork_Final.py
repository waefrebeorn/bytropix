# Wubu_Clockwork_Sleep.py
#
# THE TRI-CAMERAL MIND: REM SLEEP
#
# Tuning: Aggressive Integral Windup on Reuptake PID.
# Goal: Force the system to purge the Soul completely, 
#       driving the Body to fully adapt to the new reality.

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import chex
import numpy as np
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ... (Geodesic Engine is same as previous) ...
class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float, gear_ratio: float = 50.0) -> optax.GradientTransformation:
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
        amplified_updates = jax.tree_util.tree_map(lambda g: g * gear_ratio, updates)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified_updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified_updates, quotients)
        
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        body_updates = jax.tree_util.tree_map(lambda r: r / gear_ratio, remainders)
        new_moment1 = optax.incremental_update(body_updates, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        final_updates = jax.tree_util.tree_map(lambda m1, m2: -learning_rate * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        return final_updates, GeodesicState(count, new_moment1, new_moment2, new_topology, new_residue)
    return optax.GradientTransformation(init_fn, update_fn)

# ... (PIDSensitivity same) ...
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
        self.val = np.clip(self.val + adj * 0.05, 0.0, 1.0)
        return self.val

class PIDReuptake:
    """Aggressive Homeostasis."""
    def __init__(self):
        # Super-High Integral Gain to force long-term clearing
        self.kp, self.ki, self.kd = 2.0, 5.0, 0.1 
        self.integral = 0.0
        self.last_mag = 0.0
        self.val = 0.0
    
    def update(self, soul_magnitude):
        error = soul_magnitude - 0.0 # Goal is ZERO
        
        # Massive Windup Allowed
        self.integral = np.clip(self.integral + error, -100.0, 100.0) 
        d = error - self.last_mag
        
        adj = (self.kp * error) + (self.ki * self.integral) + (self.kd * d)
        self.last_mag = error
        
        # Can go up to 0.8 (80% transfer per step)
        self.val = np.clip(adj * 0.01, 0.0, 0.8) 
        return self.val

# ... (QLearner same) ...
class QLearnerLR:
    def __init__(self):
        self.lr = 0.05; self.table = np.zeros((10, 3)); self.state = 0; self.action = 1
    def step(self, loss):
        reward = 1.0 / (loss + 1e-5)
        next_state = min(int(loss * 20), 9)
        target = reward + 0.9 * np.max(self.table[next_state])
        self.table[self.state, self.action] += 0.1 * (target - self.table[self.state, self.action])
        if np.random.rand() < 0.1: action = np.random.randint(0, 3)
        else: action = np.argmax(self.table[next_state])
        if action == 0: self.lr *= 0.9
        elif action == 2: self.lr *= 1.1
        self.lr = np.clip(self.lr, 0.005, 0.2) 
        self.state = next_state; self.action = action
        return self.lr

def geodesic_neuron_forward(params, opt_state, x_input, sensitivity, gear_ratio):
    W_body = params['W']
    soul_W = opt_state.stored_topology['W']
    echo_W = opt_state.stored_residue['W']
    W_history = ((soul_W * (2 * jnp.pi)) + echo_W) / gear_ratio
    W_effective = W_body - (sensitivity * W_history)
    return jnp.dot(x_input, W_effective)

def run_clockwork_sleep():
    print("\n" + "="*80)
    print("THE CLOCKWORK PERCEPTRON: AGGRESSIVE CONSOLIDATION")
    print("Tuning: High Integral Gain on Reuptake PID.")
    print("Goal: Force Body to rotate 180 degrees.")
    print("="*80)

    TOTAL_STEPS = 100
    params = {'W': jnp.array([[0.0], [1.0]], dtype=jnp.float64)} 
    GEAR_RATIO = 50.0
    
    opt = geodesic_optimizer(learning_rate=1.0, gear_ratio=GEAR_RATIO)
    state = opt.init(params)
    
    pid_sens = PIDSensitivity()
    pid_reuptake = PIDReuptake()
    q_agent = QLearnerLR()
    
    print(f"{'STEP':<4} | {'ANGLE':<5} | {'BODY VECTOR':<16} | {'SOUL (Mag)':<10} | {'REUPTAKE':<8} | {'STATUS'}")
    print("-" * 80)

    for t in range(TOTAL_STEPS):
        angle_rad = jnp.pi/2 + (t / TOTAL_STEPS) * jnp.pi 
        true_vector = jnp.array([jnp.cos(angle_rad), jnp.sin(angle_rad)])
        
        key = jax.random.PRNGKey(t)
        ang = jax.random.uniform(key, minval=0, maxval=2*jnp.pi)
        x_input = jnp.array([jnp.cos(ang), jnp.sin(ang)])
        y_target = jnp.dot(x_input, true_vector)

        # Executives
        sens = pid_sens.val
        lr = q_agent.lr
        
        # Inference & Loss
        x_in_mat = x_input.reshape(1, 2) 
        y_pred = geodesic_neuron_forward(params, state, x_in_mat, sens, GEAR_RATIO)[0,0]
        loss = (y_pred - y_target)**2
        
        # Updates
        pid_sens.update(float(loss))
        lr = q_agent.step(float(loss))
        
        error = y_pred - y_target
        grad_W = error * x_input.reshape(2, 1)
        grads = {'W': grad_W}
        
        updates, state = opt.update(grads, state, params)
        scaled_updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
        params = optax.apply_updates(params, scaled_updates)
        
        # Reuptake Loop (Aggressive)
        soul_W = state.stored_topology['W']
        echo_W = state.stored_residue['W']
        total_history_W = ((soul_W * (2 * jnp.pi)) + echo_W) / GEAR_RATIO
        soul_magnitude = float(jnp.linalg.norm(total_history_W))
        
        reuptake_rate = pid_reuptake.update(soul_magnitude)
        
        transfer_W = reuptake_rate * total_history_W
        # Subtract because history accumulates Error Direction
        new_body_W = params['W'] - transfer_W
        params = {'W': new_body_W}
        
        decay_factor = 1.0 - reuptake_rate
        new_topo = jax.tree_util.tree_map(lambda s: (s * decay_factor).astype(jnp.int64), state.stored_topology)
        new_res = jax.tree_util.tree_map(lambda r: r * decay_factor, state.stored_residue)
        state = GeodesicState(state.count, state.moment1, state.moment2, new_topo, new_res)

        if t % 10 == 0 or t == TOTAL_STEPS - 1:
            w = params['W'].flatten()
            body_str = f"[{w[0]:.2f}, {w[1]:.2f}]"
            
            align = jnp.dot(w, true_vector)
            status = "✅" if align > 0.9 else "⚠️"
            if align < 0.5: status = "❌"
            
            deg = int(jnp.degrees(angle_rad))
            print(f"{t:<4} | {deg:<5} | {body_str:<16} | {soul_magnitude:<10.2f} | {reuptake_rate:<8.4f} | {status}")

    print("-" * 80)
    w_final = params['W'].flatten()
    print(f"Final Body: {w_final}")
    print(f"True Down : [0.0, -1.0]")
    
    if w_final[1] < -0.9:
        print(">>> SUCCESS: The Body fully rotated.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_clockwork_sleep()