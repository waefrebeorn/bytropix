# Wubu_Geodesic_Active_Recall.py
#
# GEODESIC AI: THE ACTIVE BRIDGE
#
# Principle: The "Forward Pass" reads the "Optimizer State".
# The Optimizer is no longer just a training tool; it is the Memory Bank.
#
# Task: Infinite Accumulation.
# The AI must output the running sum of a data stream.
# It has NO hidden state and NO weight updates. 
# It only has Topological Winding.

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. THE ENGINE (Verified Symmetric)
# ==============================================================================

class DecomposedGradient(NamedTuple):
    remainders: optax.Updates
    quotients: optax.Updates

def decompose_gradient_symmetric(updates: optax.Updates) -> DecomposedGradient:
    boundary = 2 * jnp.pi
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
    stored_topology: optax.Updates # The Soul
    stored_residue: optax.Updates  # The Echo

def geodesic_memory_cell(learning_rate: float = 0.0) -> optax.GradientTransformation:
    # Learning rate is 0.0 because we don't want to change the Weight.
    # We only want to update the Memory (Topology).
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
        
        # 1. WRITE TO BEACH (Update Memory)
        new_topology = jax.tree_util.tree_map(
            lambda acc, q: acc + q, state.stored_topology, decomposed.quotients
        )
        new_residue = jax.tree_util.tree_map(
            lambda acc, r: acc + r, state.stored_residue, decomposed.remainders
        )

        # 2. FREEZE WEIGHTS (Learning Rate 0 effectively)
        # We return Zero Updates for the weights. The intelligence is all in the state.
        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
        
        return zero_updates, GeodesicState(
            count=state.count + 1, 
            moment1=state.moment1, 
            moment2=state.moment2, 
            stored_topology=new_topology, 
            stored_residue=new_residue
        )
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# 2. THE GEODESIC NEURON (Reading from the Beach)
# ==============================================================================

def geodesic_forward_pass(params, opt_state, x_input):
    """
    The Forward Pass doesn't just look at params. 
    It looks at opt_state (The Beach).
    
    y = (Weight * x) + (Total_Memory * Read_Key)
    """
    # Retrieve Memory
    soul = opt_state.stored_topology['memory_cell']
    echo = opt_state.stored_residue['memory_cell']
    
    # Reconstruct Total Energy Stored
    total_memory = (soul * (2 * jnp.pi)) + echo
    
    # The output is the memory itself (Simulating a Recall task)
    return total_memory

# ==============================================================================
# 3. THE EXPERIMENT: INFINITE STREAM ACCUMULATION
# ==============================================================================

def run_active_recall_test():
    print("\n" + "="*60)
    print("GEODESIC AI: ACTIVE RECALL TEST")
    print("Scenario: A stream of 100 random numbers.")
    print("Task: At every step, output the SUM of all previous numbers.")
    print("Constraint: The Neural Weight is frozen at 0.0.")
    print("Mechanism: The Intelligence is entirely Topological.")
    print("="*60)

    # 1. Setup
    # 'memory_cell' is a dummy parameter. We won't actually change its value.
    # It serves as the "Address" for the topological storage.
    params = {'memory_cell': jnp.array(0.0, dtype=jnp.float64)}
    
    # We use our special memory optimizer
    opt = geodesic_memory_cell(learning_rate=0.0)
    opt_state = opt.init(params)

    # 2. The Stream
    # Let's create a sequence of inputs.
    key = jax.random.PRNGKey(42)
    stream = jax.random.uniform(key, (10,), minval=1.0, maxval=10.0)
    
    current_true_sum = 0.0
    
    print(f"{'STEP':<5} | {'INPUT':<10} | {'TRUE SUM':<12} | {'AI MEMORY (OUTPUT)':<20} | {'STATUS'}")
    print("-" * 75)

    for i, x in enumerate(stream):
        # --- STEP A: INFERENCE (READ) ---
        # The AI predicts what the current sum is BEFORE seeing the new x
        # (In this specific setup, we want it to report what it has stored so far)
        current_memory_readout = geodesic_forward_pass(params, opt_state, None)
        
        # --- STEP B: ACTION (WRITE) ---
        # We inject the new x into the memory.
        # The "Gradient" is simply the data we want to store.
        grads = {'memory_cell': jnp.array(x)}
        
        updates, opt_state = opt.update(grads, opt_state, params)
        # Note: We do NOT update params. Params stay 0.0.
        
        # --- VERIFICATION ---
        # Update our truth tracker
        prev_sum = current_true_sum
        current_true_sum += x
        
        # Check if the AI correctly remembered the previous total
        diff = jnp.abs(current_memory_readout - prev_sum)
        status = "✅" if diff < 1e-8 else "❌"
        
        print(f"{i:<5} | {x:<10.4f} | {prev_sum:<12.4f} | {current_memory_readout:<20.4f} | {status}")

    print("-" * 75)
    
    # Final check
    final_readout = geodesic_forward_pass(params, opt_state, None)
    print(f"\nFinal Memory State : {final_readout:.6f}")
    print(f"True Stream Sum    : {current_true_sum:.6f}")
    
    if jnp.abs(final_readout - current_true_sum) < 1e-8:
        print("\n>>> SUCCESS: The AI constructs its reality solely from the Beach.")
        print(">>> The Weight Matrix is empty. The Topology is full.")
    else:
        print("\n>>> FAILURE.")

if __name__ == "__main__":
    run_active_recall_test()