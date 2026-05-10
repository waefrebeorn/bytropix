# Math: The Machinery Behind the Geometry

This folder distills the mathematical foundations that power every component of the WuBu Nesting framework. These are not textbook introductions — they are the specific mathematical structures we use, with notation and conventions matching the code.

## Files

### [wubu-formalism.md](./wubu-formalism.md)
The Axiomatic-Emerent Theory's central equation:

```
Q = Σ_k q_k ∏_i α_i^{E_{k,i}}
```

A calculus for decomposing physical constants into irreducible *Foundational Atoms* (dimensionless constants like α) and *Derived Molecules*. Originally `Q = Σ_k q_k Π α_i^E.MD`.

### Hyperbolic Geometry (in `ENCODERS/hamilton-encoder-cpu/`)
The Poincaré ball model, exponential/logarithmic maps, gyrovector operations — implemented in `wubu_nesting_impl.py` (lines 12-112) and `WuBuMindJAX.py`.

Key formulas used everywhere:
```
exp_0^c(v) = tanh(√c·‖v‖ / 2) · v / (√c·‖v‖)
log_0^c(y) = atanh(√c·‖y‖) · y / (√c·‖y‖)
```

### Quaternion Rotations (in `ENCODERS/hamilton-encoder-cpu/`)
The Hamilton product — implemented in `wubu_nesting_impl.py` (lines 117-149). Used for SO(4) rotations in tangent space transitions between nested hyperbolic levels.

### BSP Trees (in `LLAMA-CPP-INTEGRATION/bsp-tree-cuda/`)
Binary Space Partitioning using quaternion-split — each node represents a rotation in SO(4), splitting a spherical cap into two child caps. The persistent pool allocation and iterative traversal pattern.

## How to Use This

1. Start with `wubu-formalism.md` — it's the most original math
2. Read the `HyperbolicUtils` class in `wubu_nesting_impl.py` for the actual implementation
3. The `hamilton_product` function is the rotational engine
4. Skip to `LLAMA-CPP-INTEGRATION/` to see how this math runs on GPU
