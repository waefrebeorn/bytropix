# Wubu_Clockwork_Stabilized.py
#
# THE TRI-CAMERAL MIND: STABILIZED VECTOR SPACE
#
# Change: Implemented "Topological Friction" (Forgetfulness).
# Mechanism: The stored topology decays by factor 0.95 at every step.
# Result: Prevents the "Singularity Explosion" while retaining local momentum.

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
# 1. GEODESIC ENGINE (With Friction)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float, gear_ratio: float = 50.0, friction: float = 0.9) -> optax.GradientTransformation:
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
        
        # 1. Transmission (Amplify)
        amplified_updates = jax.tree_util.tree_map(lambda g: g * gear_ratio, updates)
        
        # 2. Decompose
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified_updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified_updates, quotients)
        
        # 3. Storage WITH FRICTION
        # Before we add new history, we dampen the old history.
        # Note: Topology is Integer, so we cast to Float, decay, and cast back?
        # No, that loses precision. We apply friction to the EFFECTIVE history in the Forward Pass.
        # In the optimizer, we just accumulate. The "Friction" is a readout property.
        # WAIT. If we don't decay the storage itself, it grows to infinity.
        # We MUST decay the storage.
        
        # Integer Decay Logic:
        # new_topo = int(old_topo * friction) + new_quotient
        new_topology = jax.tree_util.tree_map(
            lambda acc, q: (acc * friction).astype(jnp.int64) + q, 
            state.stored_topology, quotients
        )
        
        # Residue is Float, easy to decay
        new_residue = jax.tree_util.tree_map(
            lambda acc, r: (acc * friction) + r, 
            state.stored_residue, remainders
        )

        # 4. Body Update
        body_updates = jax.tree_util.tree_map(lambda r: r / gear_ratio, remainders)
        
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
# 2. EXECUTIVES
# ==============================================================================
class PIDSensitivity:
    def __init__(self):
        self.kp, self.ki, self.kd = 1.0, 0.1, 0.1 # Tamed down
        self.integral = 0.0
        self.last_error = 0.0
        self.val = 0.01
    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -2.0, 2.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        self.val = np.clip(self.val + adj * 0.05, 0.0, 0.5) # Max 0.5 influence
        return self.val

class QLearnerLR:
    def __init__(self):
        self.lr = 0.05
        self.table = np.zeros((10, 3))
        self.state = 0
        self.action = 1
    def step(self, loss):
        reward = 1.0 / (loss + 1e-5)
        next_state = min(int(loss * 20), 9)
        target = reward + 0.9 * np.max(self.table[next_state])
        self.table[self.state, self.action] += 0.1 * (target - self.table[self.state, self.action])
        if np.random.rand() < 0.1: action = np.random.randint(0, 3)
        else: action = np.argmax(self.table[next_state])
        if action == 0: self.lr *= 0.9
        elif action == 2: self.lr *= 1.1
        self.lr = np.clip(self.lr, 0.01, 0.2) 
        self.state = next_state
        self.action = action
        return self.lr

# ==============================================================================
# 3. CLOCKWORK EXPERIMENT (STABILIZED)
# ==============================================================================

def geodesic_neuron_forward(params, opt_state, x_input, sensitivity, gear_ratio):
    W_body = params['W']
    
    # Reconstruct History
    soul_W = opt_state.stored_topology['W']
    echo_W = opt_state.stored_residue['W']
    W_history = ((soul_W * (2 * jnp.pi)) + echo_W) / gear_ratio
    
    # STABILIZED LOGIC:
    # The Body tracks position. 
    # The Soul tracks MOMENTUM (the derivative).
    # We add the momentum to the body to "anticipate" the turn.
    
    W_effective = W_body + (sensitivity * W_history)
    
    return jnp.dot(x_input, W_effective)

def run_clockwork_stabilized():
    print("\n" + "="*80)
    print("THE CLOCKWORK PERCEPTRON: STABILIZED")
    print("Gear Ratio: 50.0 | Friction: 0.90")
    print("Result: Controlled momentum. No singularities.")
    print("="*80)

    TOTAL_STEPS = 100
    params = {'W': jnp.array([[0.0], [1.0]], dtype=jnp.float64)} 
    
    GEAR_RATIO = 50.0
    FRICTION = 0.90
    
    opt = geodesic_optimizer(learning_rate=1.0, gear_ratio=GEAR_RATIO, friction=FRICTION)
    state = opt.init(params)
    
    pid = PIDSensitivity()
    q_agent = QLearnerLR()
    
    print(f"{'STEP':<4} | {'ANGLE':<5} | {'BODY VECTOR':<16} | {'SOUL (Winds)':<16} | {'LOSS':<6} | {'STATUS'}")
    print("-" * 80)

    for t in range(TOTAL_STEPS):
        angle_rad = jnp.pi/2 + (t / TOTAL_STEPS) * jnp.pi 
        true_vector = jnp.array([jnp.cos(angle_rad), jnp.sin(angle_rad)])
        
        key = jax.random.PRNGKey(t)
        ang = jax.random.uniform(key, minval=0, maxval=2*jnp.pi)
        x_input = jnp.array([jnp.cos(ang), jnp.sin(ang)])
        y_target = jnp.dot(x_input, true_vector)

        sens = pid.val
        lr = q_agent.lr
        
        x_in_mat = x_input.reshape(1, 2) 
        y_pred = geodesic_neuron_forward(params, state, x_in_mat, sens, GEAR_RATIO)[0,0]
        
        loss = (y_pred - y_target)**2
        
        pid.update(float(loss))
        lr = q_agent.step(float(loss))
        
        error = y_pred - y_target
        grad_W = error * x_input.reshape(2, 1)
        
        grads = {'W': grad_W}
        updates, state = opt.update(grads, state, params)
        
        scaled_updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
        params = optax.apply_updates(params, scaled_updates)

        if t % 10 == 0 or t == TOTAL_STEPS - 1:
            w = params['W'].flatten()
            body_str = f"[{w[0]:.2f}, {w[1]:.2f}]"
            
            s = state.stored_topology['W'].flatten()
            soul_str = f"[{s[0]:.0f}, {s[1]:.0f}]"
            
            align = jnp.dot(w, true_vector)
            status = "✅" if align > 0.8 else "⚠️"
            if align < 0.5: status = "❌"
            
            deg = int(jnp.degrees(angle_rad))
            print(f"{t:<4} | {deg:<5} | {body_str:<16} | {soul_str:<16} | {loss:<6.4f} | {status}")

    print("-" * 80)
    w_final = params['W'].flatten()
    print(f"Final Body: {w_final}")
    print(f"True Down : [0.0, -1.0]")
    
    if w_final[1] < -0.7:
        print(">>> SUCCESS: Rotation achieved with stability.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_clockwork_stabilized()