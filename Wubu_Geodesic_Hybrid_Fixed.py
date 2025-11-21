# Wubu_Geodesic_Hybrid_Fixed.py
#
# GEODESIC AI: THE BICAMERAL PERCEPTRON [CORRECTED]
#
# Changelog:
# - Fixed Optimizer Direction (Ascent -> Descent).
# - Tuned "Fatigue Sensitivity" (Alpha) for proper Integral Control response.
#
# Principle:
# The Geodesic Neuron acts as a PI Controller (Proportional-Integral).
# Weight = Proportional (Static response to input).
# Topology = Integral (Accumulated history of error/effort).

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE SYMMETRIC GEODESIC ENGINE
# ==============================================================================

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float = 0.1) -> optax.GradientTransformation:
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
        
        # 1. Decompose (Symmetric)
        quotients = jax.tree_util.tree_map(lambda g: jnp.round(g / boundary).astype(jnp.int64), updates)
        remainders = jax.tree_util.tree_map(lambda g, q: g - (q * boundary), updates, quotients)
        
        # 2. Memory Storage (Integral)
        new_topology = jax.tree_util.tree_map(lambda acc, q: acc + q, state.stored_topology, quotients)
        new_residue = jax.tree_util.tree_map(lambda acc, r: acc + r, state.stored_residue, remainders)

        # 3. Weight Update (Descent)
        # We use the 'remainders' for stability, but we MUST negate the learning rate for descent.
        new_moment1 = optax.incremental_update(remainders, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(jax.tree_util.tree_map(jnp.square, updates), state.moment2, 0.999)
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        # FIX: Negative Sign for Gradient Descent
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
# 2. THE HYBRID NEURON
# ==============================================================================

def geodesic_forward(params, opt_state, x, sensitivity):
    # Body Output (Standard)
    w = params['w']
    body = w * x
    
    # Soul Output (History)
    # Total History = (Integer Winds * 2pi) + Float Residue
    soul = opt_state.stored_topology['w']
    echo = opt_state.stored_residue['w']
    history = (soul * (2 * jnp.pi)) + echo
    
    # The "Fatigue" Equation:
    # Output = Body - (Sensitivity * History)
    return body - (sensitivity * history)

# ==============================================================================
# 3. THE TIRED RUNNER (RE-RUN)
# ==============================================================================

def run_tired_runner():
    print("\n" + "="*60)
    print("GEODESIC EXPERIMENT: THE TIRED RUNNER (FIXED)")
    print("Task: Decaying Output (1.0 -> 0.0) given Constant Input (1.0).")
    print("="*60)

    STEPS = 20
    inputs = jnp.ones(STEPS)
    targets = jnp.linspace(1.0, 0.0, STEPS)

    # --- GEODESIC AI ---
    print("\n[Geodesic AI]")
    params = {'w': jnp.array(1.0)} 
    opt = geodesic_optimizer(learning_rate=0.01) # Keep weights mostly stable
    state = opt.init(params)
    
    # Tuning: 
    # Target drops 1.0 over 20 steps. 
    # History accumulates error.
    # We need Alpha to be strong enough to convert that error into a -1.0 correction.
    FATIGUE_RATE = 0.25 

    print(f"{'STEP':<5} | {'TARGET':<8} | {'PRED':<8} | {'WEIGHT':<10} | {'HISTORY'}")
    print("-" * 60)

    for i in range(STEPS):
        x = inputs[i]
        y_target = targets[i]
        
        # 1. Inference
        y_pred = geodesic_forward(params, state, x, sensitivity=FATIGUE_RATE)
        
        # 2. Loss (Gradient = Error)
        # If Pred > Target (Too fast), Error is Positive.
        # Positive Error -> Adds to History -> Increases Fatigue term.
        error = y_pred - y_target
        grads = {'w': jnp.array(error)}
        
        # 3. Update
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        
        hist = (state.stored_topology['w'] * 2 * jnp.pi) + state.stored_residue['w']
        print(f"{i:<5} | {y_target:<8.2f} | {y_pred:<8.2f} | {params['w']:<10.4f} | {hist:<10.4f}")

    final_error = abs(y_pred - y_target)
    print("-" * 60)
    print(f"Final Prediction: {y_pred:.4f} (Target: {y_target:.4f})")
    
    if final_error < 0.05:
        print(">>> SUCCESS: The Soul accumulated fatigue and slowed the runner down.")
    else:
        print(">>> FAILURE: Tuning required.")

if __name__ == "__main__":
    run_tired_runner()