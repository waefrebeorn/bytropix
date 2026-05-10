# Wubu_Geodesic_Layer_Final.py
#
# GEODESIC LAYER v1.1 (STABILIZED)
#
# Changelog:
# - Added Input Normalization (Tanh) to prevent lever-arm explosions.
# - Capped Gradient Norms (Gradient Clipping) to 1.0.
# - This ensures the "Soul" accumulates signal, not noise spikes.

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ... (Managers are same as before) ...
class PIDSensitivity:
    def __init__(self):
        self.kp, self.ki, self.kd = 0.5, 0.1, 0.01 # Much gentler
        self.integral = 0.0; self.last_error = 0.0; self.val = 0.01
    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -2.0, 2.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        self.val = np.clip(self.val + adj * 0.01, 0.0, 0.5) # Cap sensitivity at 0.5
        return self.val

class PIDReuptake:
    def __init__(self):
        self.kp, self.ki, self.kd = 1.0, 2.0, 0.1 
        self.integral = 0.0; self.last_mag = 0.0; self.val = 0.0
    def update(self, magnitude):
        error = magnitude - 0.0 
        self.integral = np.clip(self.integral + error, -50.0, 50.0)
        d = error - self.last_mag
        adj = (self.kp * error) + (self.ki * self.integral) + (self.kd * d)
        self.last_mag = error
        self.val = np.clip(adj * 0.001, 0.0, 0.5) 
        return self.val

class QLearnerLR:
    def __init__(self):
        self.lr = 0.01; self.table = np.zeros((10, 3)); self.state = 0; self.action = 1
    def step(self, loss):
        reward = 1.0 / (loss + 1e-5)
        next_state = min(int(loss * 20), 9)
        target = reward + 0.9 * np.max(self.table[next_state])
        self.table[self.state, self.action] += 0.1 * (target - self.table[self.state, self.action])
        if np.random.rand() < 0.1: action = np.random.randint(0, 3)
        else: action = np.argmax(self.table[next_state])
        if action == 0: self.lr *= 0.9
        elif action == 2: self.lr *= 1.1
        self.lr = np.clip(self.lr, 0.001, 0.1) 
        self.state = next_state; self.action = action
        return self.lr

# ... (Optimizer is same) ...
class GeodesicState(NamedTuple):
    count: int; moment1: optax.Updates; moment2: optax.Updates
    stored_topology: optax.Updates; stored_residue: optax.Updates 

def geodesic_opt_update(updates, state, learning_rate, friction, gear_ratio):
    boundary = 2 * jnp.pi
    # Gradient Clipping (CRITICAL FIX)
    updates = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), updates)
    
    amplified = jax.tree_util.tree_map(lambda g: g * gear_ratio, updates)
    quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
    remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
    
    new_topology = jax.tree_util.tree_map(lambda acc, q: (acc * friction).astype(jnp.int64) + q, state.stored_topology, quotients)
    new_residue = jax.tree_util.tree_map(lambda acc, r: (acc * friction) + r, state.stored_residue, remainders)

    body_updates = jax.tree_util.tree_map(lambda r: r / gear_ratio, remainders)
    new_m1 = optax.incremental_update(body_updates, state.moment1, 0.9)
    new_m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
    count = state.count + 1
    m1_hat = optax.bias_correction(new_m1, 0.9, count)
    m2_hat = optax.bias_correction(new_m2, 0.999, count)
    
    final_updates = jax.tree_util.tree_map(lambda m1, m2: -learning_rate * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
    return final_updates, GeodesicState(count, new_m1, new_m2, new_topology, new_residue)

# ... (Geodesic Dense) ...
class GeodesicDense:
    def __init__(self, in_dim, out_dim, name="layer"):
        self.name = name
        self.params = {
            'w': jax.random.normal(jax.random.PRNGKey(0), (in_dim, out_dim)) * 0.1,
            'b': jnp.zeros((out_dim,))
        }
        self.opt_state = GeodesicState(
            count=jnp.array(0),
            moment1=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            moment2=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            stored_topology=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            stored_residue=jax.tree_util.tree_map(jnp.zeros_like, self.params)
        )
        self.pid_sens = PIDSensitivity()
        self.pid_reuptake = PIDReuptake()
        self.q_lr = QLearnerLR()
        self.gear_ratio = 50.0
        self.friction = 0.95

    def forward(self, x):
        # INPUT NORMALIZATION (The Retina)
        # Squash input to [-1, 1] to prevent explosion
        x_norm = jnp.tanh(x) 
        
        w_body = self.params['w']
        b_body = self.params['b']
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        w_history = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        
        sensitivity = self.pid_sens.val
        w_effective = w_body - (sensitivity * w_history)
        
        return jnp.dot(x_norm, w_effective) + b_body

    def backward(self, x, grad_output, loss_val):
        # Use normalized x for gradient calculation too
        x_norm = jnp.tanh(x)
        
        self.pid_sens.update(float(loss_val))
        lr = self.q_lr.step(float(loss_val))
        
        grad_w = jnp.dot(x_norm.T, grad_output)
        grad_b = jnp.sum(grad_output, axis=0)
        grads = {'w': grad_w, 'b': grad_b}
        
        updates, self.opt_state = geodesic_opt_update(
            grads, self.opt_state, learning_rate=lr, 
            friction=self.friction, gear_ratio=self.gear_ratio
        )
        self.params = optax.apply_updates(self.params, updates)
        
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        history_w = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        soul_mag = float(jnp.linalg.norm(history_w))
        reuptake_rate = self.pid_reuptake.update(soul_mag)
        
        transfer = reuptake_rate * history_w
        self.params['w'] = self.params['w'] - transfer
        
        decay = 1.0 - reuptake_rate
        new_topo = jax.tree_util.tree_map(lambda s: (s * decay).astype(jnp.int64), self.opt_state.stored_topology)
        new_res = jax.tree_util.tree_map(lambda r: r * decay, self.opt_state.stored_residue)
        
        self.opt_state = GeodesicState(self.opt_state.count, self.opt_state.moment1, self.opt_state.moment2, new_topo, new_res)
        return reuptake_rate

def run_mood_swing():
    print("\n" + "="*80)
    print("PRODUCT TEST: GEODESIC DENSE LAYER v1.1")
    print("Stabilized with Input Normalization & Gradient Clipping.")
    print("="*80)

    layer = GeodesicDense(in_dim=1, out_dim=1)
    STEPS = 100
    print(f"{'STEP':<4} | {'INPUT':<6} | {'TARGET':<6} | {'PRED':<6} | {'LOSS':<6} | {'REUPTAKE':<8} | {'STATUS'}")
    print("-" * 80)

    for t in range(STEPS):
        x_val = (t / 10.0) 
        if t < 50: y_target = jnp.sin(x_val)
        else: y_target = -jnp.sin(x_val)
            
        x_in = jnp.array([[x_val]])
        target_arr = jnp.array([[y_target]])
        
        y_pred = layer.forward(x_in)
        loss = jnp.mean((y_pred - target_arr)**2)
        grad_output = 2.0 * (y_pred - target_arr)
        
        reuptake = layer.backward(x_in, grad_output, loss)
        
        if t % 5 == 0:
            pred_val = float(y_pred[0,0])
            tgt_val = float(y_target)
            loss_val = float(loss)
            status = "✅" if loss_val < 0.2 else "⚠️"
            if loss_val > 0.5: status = "❌"
            if t == 50: print(f"-"*80 + "\n>>> REALITY INVERSION EVENT <<<\n" + "-"*80)
            print(f"{t:<4} | {x_val:<6.2f} | {tgt_val:<6.2f} | {pred_val:<6.2f} | {loss_val:<6.4f} | {reuptake:<8.4f} | {status}")

    print("-" * 80)
    if loss_val < 0.2: print(">>> SUCCESS.")
    else: print(">>> FAILURE.")

if __name__ == "__main__":
    run_mood_swing()