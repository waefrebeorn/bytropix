# Wubu_Deep_Geodesic_Batch.py
#
# THE GEODESIC MLP [SCALED & BATCHED]
#
# Changes:
# 1. Mini-Batching (BS=16): Stabilizes the Topological Winding.
# 2. Wider Architecture: 32 Neurons per layer.
# 3. Gentler Reuptake: PID gains tuned down to prevent weight collapse.
#
# Goal: Solve the Two-Spirals problem.

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. MANAGERS (Tuned for Batches)
# ==============================================================================
class PIDSensitivity:
    def __init__(self):
        self.kp, self.ki, self.kd = 0.5, 0.1, 0.01 
        self.integral = 0.0; self.last_error = 0.0; self.val = 0.01
    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -5.0, 5.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        self.val = np.clip(self.val + adj * 0.01, 0.0, 0.5)
        return self.val

class PIDReuptake:
    def __init__(self):
        # Lower gains to prevent "Seizure Mode"
        self.kp, self.ki, self.kd = 0.5, 0.5, 0.05 
        self.integral = 0.0; self.last_mag = 0.0; self.val = 0.0
    def update(self, magnitude):
        error = magnitude - 0.0 
        self.integral = np.clip(self.integral + error, -50.0, 50.0)
        d = error - self.last_mag
        adj = (self.kp * error) + (self.ki * self.integral) + (self.kd * d)
        self.last_mag = error
        # Cap at 10% transfer per batch to be safe
        self.val = np.clip(adj * 0.0005, 0.0, 0.1) 
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

# ==============================================================================
# 2. GEODESIC ENGINE (Lossless)
# ==============================================================================
class GeodesicState(NamedTuple):
    count: int; moment1: optax.Updates; moment2: optax.Updates
    stored_topology: optax.Updates; stored_residue: optax.Updates 

def geodesic_opt_update(updates, state, learning_rate, friction, gear_ratio):
    boundary = 2 * jnp.pi
    # No Clipping (Lossless)
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

class GeodesicDense:
    def __init__(self, in_dim, out_dim, name="layer"):
        self.name = name
        # Initialize weights larger for deep nets
        self.params = {'w': jax.random.normal(jax.random.PRNGKey(0), (in_dim, out_dim)) * 0.5, 'b': jnp.zeros((out_dim,))}
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
        self.gear_ratio = 50.0; self.friction = 0.95
        self.last_input = None

    def forward(self, x):
        self.last_input = jnp.tanh(x) 
        w_body = self.params['w']
        b_body = self.params['b']
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        w_history = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        sensitivity = self.pid_sens.val
        w_effective = w_body - (sensitivity * w_history)
        return jnp.dot(self.last_input, w_effective) + b_body

    def backward(self, grad_output, loss_val):
        grad_w = jnp.dot(self.last_input.T, grad_output)
        grad_b = jnp.sum(grad_output, axis=0)
        
        grad_input = jnp.dot(grad_output, self.params['w'].T)
        grad_input = grad_input * (1.0 - self.last_input**2)

        self.pid_sens.update(float(loss_val))
        lr = self.q_lr.step(float(loss_val))
        
        grads = {'w': grad_w, 'b': grad_b}
        updates, self.opt_state = geodesic_opt_update(grads, self.opt_state, lr, self.friction, self.gear_ratio)
        self.params = optax.apply_updates(self.params, updates)
        
        # Reuptake (Sleep)
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        history_w = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        reuptake_rate = self.pid_reuptake.update(float(jnp.linalg.norm(history_w)))
        
        self.params['w'] = self.params['w'] - (reuptake_rate * history_w)
        
        decay = 1.0 - reuptake_rate
        new_topo = jax.tree_util.tree_map(lambda s: (s * decay).astype(jnp.int64), self.opt_state.stored_topology)
        new_res = jax.tree_util.tree_map(lambda r: r * decay, self.opt_state.stored_residue)
        self.opt_state = GeodesicState(self.opt_state.count, self.opt_state.moment1, self.opt_state.moment2, new_topo, new_res)

        return grad_input, reuptake_rate

# ==============================================================================
# 3. EXPERIMENT
# ==============================================================================
def generate_spiral(n_points=100):
    theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi 
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(n_points,2)*0.1 # Less noise
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(n_points,2)*0.1
    res_a = np.append(x_a, np.zeros((n_points,1)), axis=1)
    res_b = np.append(x_b, np.ones((n_points,1)), axis=1)
    return np.vstack((res_a, res_b))

def run_deep_spiral():
    print("\n" + "="*80)
    print("DEEP GEODESIC NETWORK: BATCHED & SCALED")
    print("Architecture: [2] -> [32] -> [32] -> [1]")
    print("Batch Size: 16 | Tuning: Stable Reuptake")
    print("="*80)

    l1 = GeodesicDense(2, 32, "Hidden1")
    l2 = GeodesicDense(32, 32, "Hidden2")
    l3 = GeodesicDense(32, 1, "Output")
    
    data = generate_spiral(200) # 400 points
    X = jnp.array(data[:, :2])
    X = X / jnp.max(jnp.abs(X)) 
    Y = jnp.array(data[:, 2]).reshape(-1, 1)

    BATCH_SIZE = 16
    EPOCHS = 100
    
    print(f"{'EPOCH':<5} | {'LOSS':<8} | {'ACCURACY':<8} | {'L1 SLEEP':<12} | {'L3 SLEEP':<12}")
    print("-" * 70)

    for epoch in range(EPOCHS):
        total_loss = 0.0; correct = 0
        l1_r_avg = 0.0; l3_r_avg = 0.0
        n_batches = 0
        
        perm = np.random.permutation(len(X))
        X_shuff = X[perm]; Y_shuff = Y[perm]

        for i in range(0, len(X), BATCH_SIZE):
            x_curr = X_shuff[i:i+BATCH_SIZE]
            y_curr = Y_shuff[i:i+BATCH_SIZE]
            
            # Forward
            h1 = jnp.tanh(l1.forward(x_curr))
            h2 = jnp.tanh(l2.forward(h1))
            out = l3.forward(h2)
            
            prob = 1.0 / (1.0 + jnp.exp(-out))
            pred_labels = (prob > 0.5).astype(jnp.float64)
            correct += jnp.sum(pred_labels == y_curr)
            
            loss = jnp.mean((prob - y_curr)**2)
            total_loss += float(loss)

            # Backward (Batch)
            grad_out = (prob - y_curr) * prob * (1.0 - prob)
            # Normalized by batch size
            grad_out = grad_out / BATCH_SIZE 
            
            grad_h2, r3 = l3.backward(grad_out, loss)
            grad_h1, r2 = l2.backward(grad_h2, loss)
            _, r1 = l1.backward(grad_h1, loss)
            
            l1_r_avg += r1; l3_r_avg += r3
            n_batches += 1

        avg_loss = total_loss / n_batches
        acc = correct / len(X)
        l1_r_avg /= n_batches
        l3_r_avg /= n_batches

        if epoch % 10 == 0 or epoch == EPOCHS-1:
            print(f"{epoch:<5} | {avg_loss:<8.4f} | {acc:<8.4f} | {l1_r_avg:<12.4f} | {l3_r_avg:<12.4f}")

    print("-" * 70)
    if acc > 0.90: print(">>> SUCCESS: Solved the Spiral.")
    else: print(">>> FAILURE.")

if __name__ == "__main__":
    run_deep_spiral()