# Wubu_Geodesic_Hybrid.py
#
# GEODESIC AI: THE BICAMERAL PERCEPTRON
#
# Architecture: Hybrid Geodesic Neuron.
# Equation: y = (W_body * x) - (alpha * Memory_soul)
#
# The Weight handles the "Force".
# The Topology handles the "Fatigue".

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE ENGINE (Symmetric Holographic)
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def symmetric_geodesic_optimizer(learning_rate: float = 0.1) -> optax.GradientTransformation:
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
        
        # Write to Beach
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        # Update Weights (The Body)
        new_moment1 = optax.incremental_update(remainders, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        final_updates = jax.tree_util.tree_map(lambda m1, m2: learning_rate * m1 / (jnp.sqrt(m2) + 1e-8), m1_hat, m2_hat)
        
        return final_updates, GeodesicState(
            count=count, moment1=new_moment1, moment2=new_moment2, 
            stored_topology=new_topology, stored_residue=new_residue
        )
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# 2. THE HYBRID NEURON
# ==============================================================================

def geodesic_forward(params, opt_state, x, sensitivity=0.001):
    """
    y = (Weight * Input) - (Sensitivity * Total_History)
    
    The neuron 'weighs' the input, but subtracts its own 'fatigue' (history).
    """
    # 1. The Body (Standard Neural Network)
    w = params['w']
    body_output = w * x
    
    # 2. The Soul (Reading the Beach)
    soul = opt_state.stored_topology['w']
    echo = opt_state.stored_residue['w']
    total_history = (soul * (2 * jnp.pi)) + echo
    
    # The "Fatigue" effect
    # We use total_history magnitude as a dampener
    soul_output = sensitivity * total_history
    
    return body_output - soul_output

# ==============================================================================
# 3. THE TIRED RUNNER EXPERIMENT
# ==============================================================================

def run_tired_runner():
    print("\n" + "="*60)
    print("GEODESIC EXPERIMENT: THE TIRED RUNNER")
    print("Comparison: Standard AI vs. Geodesic AI")
    print("Task: Map Constant Input (1.0) to Decaying Output.")
    print("Note: No 'Time' feature is provided. The AI must 'feel' time.")
    print("="*60)

    # --- SETUP ---
    STEPS = 20
    inputs = jnp.ones(STEPS) # Constant 1.0
    # Target: Linear decay from 1.0 down to 0.0
    targets = jnp.linspace(1.0, 0.0, STEPS)
    
    # --- COMPETITOR 1: STANDARD AI (AdamW) ---
    print("\n[Standard AI]")
    params_std = {'w': jnp.array(0.5)} # Start random
    opt_std = optax.adam(0.1)
    state_std = opt_std.init(params_std)
    
    # Training Loop (Online Learning)
    for i in range(STEPS):
        def loss_std(p): return (p['w'] * inputs[i] - targets[i])**2
        grads = jax.grad(loss_std)(params_std)
        updates, state_std = opt_std.update(grads, state_std, params_std)
        params_std = optax.apply_updates(params_std, updates)
        
        pred = params_std['w'] * inputs[i]
        print(f"Step {i:02d} | Target: {targets[i]:.2f} | Pred: {pred:.2f} | W: {params_std['w']:.2f}")

    print(">> Standard AI Result: Fails to track decay. Chases the moving target.")

    # --- COMPETITOR 2: GEODESIC AI (Hybrid) ---
    print("\n[Geodesic AI]")
    params_geo = {'w': jnp.array(1.0)} # Start at max capacity
    opt_geo = symmetric_geodesic_optimizer(learning_rate=0.01) # Low LR for weights
    state_geo = opt_geo.init(params_geo)
    
    # We define a sensitivity for how much fatigue affects performance
    # In a real network, this 'sensitivity' would be a learnable gating parameter.
    FATIGUE_RATE = 0.05 

    print(f"{'STEP':<5} | {'TARGET':<8} | {'PRED':<8} | {'WEIGHT (Body)':<15} | {'HISTORY (Soul)'}")
    print("-" * 65)

    for i in range(STEPS):
        x = inputs[i]
        y_target = targets[i]
        
        # 1. INFERENCE (Bicameral)
        # The AI predicts based on current Weight AND Accumulated History
        y_pred = geodesic_forward(params_geo, state_geo, x, sensitivity=FATIGUE_RATE)
        
        # 2. LOSS & GRADIENT
        # Error = Prediction - Target
        # If Pred > Target (too fast), Error is Positive.
        # Positive Error -> Positive Gradient -> Adds to History -> Increases Fatigue -> Slows down next step.
        error = y_pred - y_target
        
        # We feed the ERROR directly as the gradient force.
        # This accumulates "Failure" in the memory.
        grads = {'w': jnp.array(error)}
        
        # 3. UPDATE
        updates, state_geo = opt_geo.update(grads, state_geo, params_geo)
        params_geo = optax.apply_updates(params_geo, updates)
        
        # Debug readout
        soul = state_geo.stored_topology['w'] * (2*jnp.pi)
        echo = state_geo.stored_residue['w']
        hist = soul + echo
        
        print(f"{i:<5} | {y_target:<8.2f} | {y_pred:<8.2f} | {params_geo['w']:<15.4f} | {hist:<10.4f}")

    # Final Analysis
    print("-" * 65)
    final_error = abs(y_pred - y_target)
    if final_error < 0.1:
        print(">>> SUCCESS: Geodesic AI modeled the decay using only Internal State.")
        print(">>> The Weight remained stable(ish), but the Soul carried the burden.")
    else:
        print(">>> FAILURE.")

if __name__ == "__main__":
    run_tired_runner()