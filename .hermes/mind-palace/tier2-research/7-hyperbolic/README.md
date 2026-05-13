# Hyperbolic Neural Networks — Theory for WuBu

Source papers: `.hermes/research/papers/research_papers/`

## Papers

| Paper | ID | Key Contribution |
|-------|----|-----------------|
| **Poincaré Embeddings** | 1705.08039 | Embedding DAGs in hyperbolic space, Riemannian optimization |
| **Hyperbolic Neural Networks** | 1802.03367 | Möbius transformation for neural layers, exp/log maps |
| **Fully Hyperbolic NNs** | 2205.04641 | Lorentz model, all layers hyperbolic (no Euclidean bottleneck) |
| **Möbius Transformers** | 2311.11394 | Full hyperbolic transformer, self-attention in Lorentz |

## Key Math for Implementation

### Poincaré Ball Model
```
B^d_c = {x ∈ R^d : c||x||² < 1}   // c = curvature (-1 for unit ball)
```

### Exp/Log Maps (from HNN paper)
```
exp_x(v) = x ⊕ (tanh(√c||v||/2) × v/(√c||v||))    // map tangent → ball
log_x(y) = 2/(√c) × artanh(√c||-x⊕y||) × (-x⊕y)/||-x⊕y||  // ball → tangent
```

### Möbius Linear Layer (from HNN paper)
```
f(x) = exp_0(W·log_0(x) + b)    // Map to tangent, linear, map back
```
Also equivalent to: `M⊗(x) = tanh(||Mx||/R) × Mx/||Mx||`

### Möbius Transformer Attention (from Möbius Transformers paper)
- Query, Key, Value in Lorentz model
- Attention score = Lorentzian inner product (which is proper distance)
- Output = Möbius combination of value vectors

## For WuBu: The Gated DeltaNet Connection

The most important insight: **Gated DeltaNet's linear recurrence**:
```
h_t = λ_t ⊙ h_{t-1} + gate_t ⊙ v_t
```

Can become a **hyperbolic recurrence**:
```
h_t = gyration(h_{t-1}, g_t ⊙ v_t, λ_t)
     = h_{t-1} ⊕ (λ_t ⊙ gyration(-h_{t-1}, g_t ⊙ v_t))
```

Where gyration replaces scaling, and ⊕ is Möbius addition.
This keeps h_t in the Poincaré ball at all times, and the recurrence is still O(n).

## Key Decision: Which Hyperbolic Model?

| Model | Pros | Cons |
|-------|------|------|
| **Poincaré Ball** | Intuitive, exp/log maps simple | Numerical instability near boundary |
| **Lorentz (Hyperboloid)** | Numerically stable, closed-form dist | More complex math, 1 extra dim |
| **Product of Poincaré** | Multiple curvatures | More params |

**Recommendation: Start with Poincaré Ball** — our WuBu math is already in this model,
the bytropix Lean proofs are in this model, and the Möbius operations are proven.
