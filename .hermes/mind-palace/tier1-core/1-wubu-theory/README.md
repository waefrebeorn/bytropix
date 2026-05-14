# WuBu Nesting Theory — Core Math

## Where To Find It

Full text: `../../THEORY/WuBu_Nesting.md` (converted from PDF, 27K chars)
Full text v0.1: `../../ENCODERS/hash-mind/WuBuNestingv0.1.md` (converted from PDF, 36K chars)

## Key Concepts for Implementation

### 1. Poincaré Ball Model
- Hyperbolic space of constant negative curvature
- Points: vectors x where ||x|| < 1 (in unit ball)
- Distance: `d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))`
- Möbius addition: `x ⊕ y = ((1+2⟨x,y⟩+||y||²)x + (1-||x||²)y) / (1+2⟨x,y⟩+||x||²||y||²)`
- **Verified with Qwen3.6 embeddings:** R=0.956 (3×mean_norm), max norm after map=0.517

### 2. Gyration
- Gyration operator: `gyr[x,y]z = -(x ⊕ y) ⊕ (x ⊕ (y ⊕ z))`
- Preserves inner product but IS NOT linear
- Key property: `x ⊕ y = gyr[x,y](y ⊕ x)` (non-commutative)
- For implementation: gyration = rotation in tangent space

### 3. Möbius Transform (for neural nets)
- Möbius matrix-vector multiplication: `M⊗(x) = tanh(||Mx||/R) × Mx/||Mx||`
- Maps Euclidean linear layer → hyperbolic
- `R` = Poincaré ball radius (trainable, but we fixed R=0.956 based on embedding analysis)

### 4. WuBu Optimization (Toroidal Gradient) — DEPRECATED FOR POINCARÉ

**Important:** The toroidal gradient (`g_wubu = g % 2π`) was designed for the K-theory /
Hopf algebra approach in the original WuBu papers. For the Poincaré ball model used here,
we need **Riemannian SGD (RSGD)**, NOT toroidal optimization:

```
RSGD: w_new = exp_map(log_map(w, R) - lr * g, R)  // step in tangent space, project back
```

The toroidal optimizer from the baseline C code is for a different mathematical framework.
Don't reuse it for this project unless we switch back to toroidal geometry.

### 5. K-Theory Connection
- Vector bundles over hyperbolic manifolds
- Nested structure = fiber bundle hierarchy
- Each layer = section of a vector bundle
- MLP = composition of bundle morphisms
- **Practical relevance:** The nested MoE tree (Phase 4) implements a fiber bundle structure
  where each level is a sub-bundle over the previous.

### 6. Hopf Algebra Structure
- Coproduct: `Δ(x) = x ⊗ 1 + 1 ⊗ x` (splitting activations)
- Antipode: `S(x) = -x` (gradient reversal)
- Counit: `ε(x) = 0` (bias)
- Enables gradient routing through nested structures
- **Practical relevance:** The MTP (multi-token prediction) head implements a coproduct-like
  split: the same hidden state predicts both next-token and next-next-token.
- Gradient reversal appears in adversarial training / routing regularization.

## What This Means For WuBuText

| Operation | What We Actually Use | Where |
|-----------|---------------------|-------|
| **exp_map** | `tanh(||x||/R) × x/||x||` | Embedding → Poincaré (Phase 1 ✅) |
| **log_map** | `R × artanh(||x||) × x/||x||` | Poincaré → LM Head (Phase 3) |
| **Möbius add** | `x ⊕ y` formula above | SSM recurrence replacement (Phase 2) |
| **Gyration** | `gyr[x,y]z` | Attention mixing (Phase 2) |
| **Poincaré distance** | `arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))` | MoE router (Phase 4) |
| **RSGD optimizer** | Riemannian SGD in tangent space | Training (Phase 3) |
| **Toroidal gradient** | `g % 2π` | NOT USED — baseline code only |
| **Hopf algebra** | Gradient splitting | MTP head (Phase 3) |

## Lean Proofs

4 verified proofs exist at `MATH/lean/wubu_proofs/`:
- `MobiusAddPreservesBall`: Möbius addition stays in Poincaré ball
- `PoincareBallTanhArtanh`: tanh∘artanh=id (exp/log maps are inverses)
- `MLACompression`: Factor identity for KV compression
- `HyperbolicGyration`: 1D gyration = identity (base case for induction)

These proofs verify the core math used in our C implementation.
