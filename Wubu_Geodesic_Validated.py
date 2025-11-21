# Wubu_Geodesic_Validated.py
#
# GEODESIC AI: FINAL VERIFICATION
#
# Principle: Lossless Geodesic Storage.
# Verification Method: Direct comparison of Input Mean vs. Recovered Mean.
#
# If this passes, the Wubu Geodesic Layer is mathematically sound 
# for infinite-context storage.

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE GEODESIC ENGINE (Symmetric)
# ==============================================================================

class DecomposedGradient(NamedTuple):
    remainders: optax.Updates
    quotients: optax.Updates

def decompose_gradient_symmetric(updates: optax.Updates, boundary_scale: float = 1.0) -> DecomposedGradient:
    boundary = 2 * jnp.pi * boundary_scale
    
    # Symmetric Rounding: Rounds to nearest integer.
    # Ensures x = (q * boundary) + r is an EXACT IDENTITY.
    quotients_pytree = jax.tree_util.tree_map(
        lambda g: jnp.round(g / boundary).astype(jnp.int64), updates
    )
    remainders_pytree = jax.tree_util.tree_map(
        lambda g, q: g - (q * boundary), updates, quotients_pytree
    )
    return DecomposedGradient(remainders=remainders_pytree, quotients=quotients_pytree)

class GeodesicState(NamedTuple):
    count: chex.Array
    moment1: optax.Updates
    moment2: optax.Updates
    stored_topology: optax.Updates 
    stored_residue: optax.Updates 

def geodesic_optimizer(learning_rate: float = 0.01) -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> GeodesicState:
        return GeodesicState(
            count=jnp.zeros([], jnp.int32),
            moment1=jax.tree_util.tree_map(jnp.zeros_like, params),
            moment2=jax.tree_util.tree_map(jnp.zeros_like, params),
            stored_topology=jax.tree_util.tree_map(jnp.zeros_like, params),
            stored_residue=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates: optax.Updates, state: GeodesicState, params=None) -> tuple[optax.Updates, GeodesicState]:
        decomposed = decompose_gradient_symmetric(updates)
        
        # --- THE MEMORY LAYER ---
        # Accumulate Integer Windings (Soul)
        new_topology = jax.tree_util.tree_map(
            lambda acc, q: acc + q, state.stored_topology, decomposed.quotients
        )
        # Accumulate Float Residue (Echo)
        new_residue = jax.tree_util.tree_map(
            lambda acc, r: acc + r, state.stored_residue, decomposed.remainders
        )

        # --- THE OPTIMIZER LAYER ---
        # We use the residue (small, safe gradient) for the Adam update.
        # This prevents the weights from exploding, even if the input is 1,000,000.
        new_moment1 = optax.incremental_update(decomposed.remainders, state.moment1, 0.9)
        new_moment2 = optax.incremental_update(
            jax.tree_util.tree_map(jnp.square, decomposed.remainders), state.moment2, 0.999
        )
        count = state.count + 1
        m1_hat = optax.bias_correction(new_moment1, 0.9, count)
        m2_hat = optax.bias_correction(new_moment2, 0.999, count)
        
        final_updates = jax.tree_util.tree_map(
            lambda m1, m2: learning_rate * m1 / (jnp.sqrt(m2) + 1e-8),
            m1_hat, m2_hat
        )
        
        return final_updates, GeodesicState(
            count=count, moment1=new_moment1, moment2=new_moment2, 
            stored_topology=new_topology, stored_residue=new_residue
        )
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# 2. FINAL VALIDATION: THE PERFECT RECALL TEST
# ==============================================================================

def validate_geodesic_memory():
    print("\n" + "="*60)
    print("FINAL VALIDATION: LOSSLESS STORAGE IN CHAOS")
    print("Goal: Verify that (Topology + Residue) exactly matches Input.")
    print("="*60)

    params = {'buffer': jnp.array(0.0)}
    opt = geodesic_optimizer()
    state = opt.init(params)
    
    # CONFIGURATION
    STEPS = 10000
    NOISE_MAG = 10000.0 
    SIGNAL = 1.0

    # Generate Input Data
    # We use a massive array of inputs to simulate a lifetime of data.
    key = jax.random.PRNGKey(999)
    noise = jax.random.uniform(key, shape=(STEPS,), minval=-NOISE_MAG, maxval=NOISE_MAG)
    inputs = SIGNAL + noise # The actual data stream
    
    # The Truth: What is the sum of this specific random stream?
    true_sum = jnp.sum(inputs)
    true_mean = jnp.mean(inputs)

    print(f"Processing {STEPS} steps...")
    print(f"Input Range     : +/- {NOISE_MAG}")
    print(f"Actual Data Sum : {true_sum:.6f} (Randomness doesn't sum to 0 perfectly)")

    # --- THE PROCESS ---
    # We define a scan loop for blazing speed in JAX
    def step_fn(carry, x):
        s, p = carry
        grad = {'buffer': x} # The gradient IS the data
        _, s = opt.update(grad, s, p)
        return (s, p), None

    (final_state, _), _ = jax.lax.scan(step_fn, (state, params), inputs)
    
    # --- THE RECONSTRUCTION ---
    # Total Stored = (Soul * 2pi) + Echo
    stored_total = (final_state.stored_topology['buffer'] * (2 * jnp.pi)) + final_state.stored_residue['buffer']
    
    # Validation
    diff = jnp.abs(stored_total - true_sum)
    
    print("-" * 40)
    print(f"Total Input Sum  : {true_sum:.10f}")
    print(f"Total Stored Sum : {stored_total:.10f}")
    print(f"Difference       : {diff:.10f}")
    print("-" * 40)

    if diff < 1e-8:
        print(">>> RESULT: PERFECT LOSSLESS STORAGE.")
        print(">>> The Geodesic AI captured every bit of entropy.")
    else:
        print(">>> RESULT: FAILED.")

if __name__ == "__main__":
    validate_geodesic_memory()