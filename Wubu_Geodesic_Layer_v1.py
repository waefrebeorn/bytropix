# Wubu_Geodesic_Layer_v1.py
#
# GEODESIC LAYER v1.0 (PRODUCTION PROTOTYPE)
#
# A drop-in, self-managing Neural Network Layer.
# It contains its own Optimizer, Memory (Soul), and Executive Managers.
#
# Features:
# - Self-Tuning Learning Rate (Q-Learner)
# - Self-Regulating Sensitivity (Amygdala PID)
# - Automatic Memory Consolidation (Hippocampus PID)

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import NamedTuple, Tuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE MANAGERS (The "Brain" of the Layer)
# ==============================================================================

class PIDSensitivity:
    def __init__(self):
        self.kp, self.ki, self.kd = 2.0, 0.5, 0.1
        self.integral = 0.0; self.last_error = 0.0; self.val = 0.01
    def update(self, error):
        err = abs(error)
        self.integral = np.clip(self.integral + err, -5.0, 5.0)
        d = err - self.last_error
        adj = (self.kp * err) + (self.ki * self.integral) + (self.kd * d)
        self.last_error = err
        self.val = np.clip(self.val + adj * 0.05, 0.0, 1.0)
        return self.val

class PIDReuptake:
    def __init__(self):
        self.kp, self.ki, self.kd = 2.0, 5.0, 0.1 
        self.integral = 0.0; self.last_mag = 0.0; self.val = 0.0
    def update(self, magnitude):
        error = magnitude - 0.0 
        self.integral = np.clip(self.integral + error, -100.0, 100.0)
        d = error - self.last_mag
        adj = (self.kp * error) + (self.ki * self.integral) + (self.kd * d)
        self.last_mag = error
        self.val = np.clip(adj * 0.01, 0.0, 0.8) 
        return self.val

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

# ==============================================================================
# 2. THE HARDWARE (Geodesic Optimizer)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: int
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_opt_update(updates, state, learning_rate, friction, gear_ratio):
    boundary = 2 * jnp.pi
    
    # Transmission
    amplified = jax.tree_util.tree_map(lambda g: g * gear_ratio, updates)
    
    # Decompose
    quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), amplified)
    remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), amplified, quotients)
    
    # Storage (With Friction)
    new_topology = jax.tree_util.tree_map(lambda acc, q: (acc * friction).astype(jnp.int64) + q, state.stored_topology, quotients)
    new_residue = jax.tree_util.tree_map(lambda acc, r: (acc * friction) + r, state.stored_residue, remainders)

    # Body Update (Descent)
    body_updates = jax.tree_util.tree_map(lambda r: r / gear_ratio, remainders)
    new_m1 = optax.incremental_update(body_updates, state.moment1, 0.9)
    new_m2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
    count = state.count + 1
    m1_hat = optax.bias_correction(new_m1, 0.9, count)
    m2_hat = optax.bias_correction(new_m2, 0.999, count)
    
    final_updates = jax.tree_util.tree_map(lambda m1, m2: -learning_rate * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
    
    return final_updates, GeodesicState(count, new_m1, new_m2, new_topology, new_residue)

# ==============================================================================
# 3. THE PRODUCT: GEODESIC DENSE LAYER
# ==============================================================================

class GeodesicDense:
    def __init__(self, in_dim, out_dim, name="layer"):
        self.name = name
        # Initialize Weights (Body)
        self.params = {
            'w': jax.random.normal(jax.random.PRNGKey(0), (in_dim, out_dim)) * 0.1,
            'b': jnp.zeros((out_dim,))
        }
        # Initialize Optimizer State (Soul)
        self.opt_state = GeodesicState(
            count=jnp.array(0),
            moment1=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            moment2=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            stored_topology=jax.tree_util.tree_map(jnp.zeros_like, self.params),
            stored_residue=jax.tree_util.tree_map(jnp.zeros_like, self.params)
        )
        # Initialize Executives
        self.pid_sens = PIDSensitivity()
        self.pid_reuptake = PIDReuptake()
        self.q_lr = QLearnerLR()
        
        self.gear_ratio = 50.0
        self.friction = 0.90 # Dynamic, but base is 0.90

    def forward(self, x):
        """
        Forward pass with Soul-Correction.
        y = x @ (W_body - Sensitivity * W_soul) + b
        """
        w_body = self.params['w']
        b_body = self.params['b']
        
        # Retrieve Soul
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        w_history = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        
        # Amygdala controls Sensitivity
        sensitivity = self.pid_sens.val
        
        # Effective Weight (Negative Feedback)
        w_effective = w_body - (sensitivity * w_history)
        
        # Note: We don't correct Bias in this POC for simplicity, but we could.
        return jnp.dot(x, w_effective) + b_body

    def backward(self, x, grad_output, loss_val):
        """
        Manages the entire update cycle: 
        Executives -> Optimizer -> Reuptake.
        """
        # 1. Update Executives
        self.pid_sens.update(float(loss_val))
        lr = self.q_lr.step(float(loss_val))
        
        # 2. Calculate Gradients (Manual Backprop for POC)
        # dL/dw = x.T @ grad_output
        grad_w = jnp.dot(x.T, grad_output)
        grad_b = jnp.sum(grad_output, axis=0)
        grads = {'w': grad_w, 'b': grad_b}
        
        # 3. Optimizer Update
        updates, self.opt_state = geodesic_opt_update(
            grads, self.opt_state, learning_rate=lr, 
            friction=self.friction, gear_ratio=self.gear_ratio
        )
        self.params = optax.apply_updates(self.params, updates)
        
        # 4. Reuptake Loop (Sleep)
        # Calculate Soul Stress
        soul_w = self.opt_state.stored_topology['w']
        echo_w = self.opt_state.stored_residue['w']
        history_w = ((soul_w * (2 * jnp.pi)) + echo_w) / self.gear_ratio
        soul_mag = float(jnp.linalg.norm(history_w))
        
        reuptake_rate = self.pid_reuptake.update(soul_mag)
        
        # Transfer Momentum
        transfer = reuptake_rate * history_w
        self.params['w'] = self.params['w'] - transfer
        
        # Decay Soul
        decay = 1.0 - reuptake_rate
        new_topo = jax.tree_util.tree_map(lambda s: (s * decay).astype(jnp.int64), self.opt_state.stored_topology)
        new_res = jax.tree_util.tree_map(lambda r: r * decay, self.opt_state.stored_residue)
        
        # Reconstruct state tuple
        self.opt_state = GeodesicState(
            self.opt_state.count, self.opt_state.moment1, self.opt_state.moment2,
            new_topo, new_res
        )
        
        return reuptake_rate # Return for logging

# ==============================================================================
# 4. EXPERIMENT: THE MOOD SWING (Deep Learning Test)
# ==============================================================================
def run_mood_swing():
    print("\n" + "="*80)
    print("PRODUCT TEST: GEODESIC DENSE LAYER v1.0")
    print("Experiment: 'The Mood Swing'")
    print("Task: Learn y=sin(x). At Step 50, target flips to y=-sin(x).")
    print("Goal: The Layer must adapt its weights dynamically.")
    print("="*80)

    # Create a Single Geodesic Layer (1 Input -> 1 Output)
    # Ideally this would be an MLP, but let's test the core unit first.
    layer = GeodesicDense(in_dim=1, out_dim=1, name="GeodesicNeuron")
    
    STEPS = 100
    print(f"{'STEP':<4} | {'INPUT':<6} | {'TARGET':<6} | {'PRED':<6} | {'LOSS':<6} | {'REUPTAKE':<8} | {'STATUS'}")
    print("-" * 80)

    for t in range(STEPS):
        # Generate Data
        # Phase 1 (0-50): y = sin(x)
        # Phase 2 (50-100): y = -sin(x) (INVERTED REALITY)
        
        x_val = (t / 10.0) # Walking along x
        
        if t < 50:
            y_target = jnp.sin(x_val)
            mode = "Phase 1"
        else:
            y_target = -jnp.sin(x_val)
            mode = "Phase 2 (FLIP)"
            
        x_in = jnp.array([[x_val]]) # Batch size 1
        target_arr = jnp.array([[y_target]])
        
        # 1. Forward
        y_pred = layer.forward(x_in)
        
        # 2. Loss
        loss = jnp.mean((y_pred - target_arr)**2)
        
        # 3. Gradient (dLoss/dPred = 2 * (Pred - Target))
        grad_output = 2.0 * (y_pred - target_arr)
        
        # 4. Backward (Internal Management)
        reuptake = layer.backward(x_in, grad_output, loss)
        
        if t % 5 == 0:
            pred_val = float(y_pred[0,0])
            tgt_val = float(y_target)
            loss_val = float(loss)
            
            status = "✅" if loss_val < 0.1 else "⚠️"
            if loss_val > 0.5: status = "❌"
            if t == 50: print(f"-"*80 + "\n>>> REALITY INVERSION EVENT <<<\n" + "-"*80)
            
            print(f"{t:<4} | {x_val:<6.2f} | {tgt_val:<6.2f} | {pred_val:<6.2f} | {loss_val:<6.4f} | {reuptake:<8.4f} | {status}")

    print("-" * 80)
    final_loss = float(loss)
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss < 0.1:
        print(">>> SUCCESS: Geodesic Layer handled the Mood Swing.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_mood_swing()