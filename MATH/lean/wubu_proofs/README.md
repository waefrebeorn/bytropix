# Wubu Proofs — Formal Verification of WuBu Nesting

This Lean 4 project formalizes the core mathematical claims
behind the WuBu Nested Hyperbolic framework.

## Proofs

| # | File | Theorem | Status |
|---|------|---------|--------|
| 1 | `PoincareBall.lean` | `exp_0^c(log_0^c(y)) = y` identity | Partial |
| 2 | `MobiusAdd.lean` | Möbius addition preserves the unit ball | 1D proven |
| 3 | `MLACompression.lean` | Low-rank KV compression error bound | Sketch |
| 4 | `HyperbolicGyration.lean` | Gyration preserves the ball | Sketch |

## Build

```bash
cd /home/wubu/bytropix/MATH/lean/wubu_proofs
lake build
```

Lake will automatically download mathlib4 as a dependency.

## TODO

1. Complete Proof 1 using `Real.tanh_arctanh` + norm cancellation
2. Extend Proof 2 to n-dimensional case (Cauchy-Schwarz)
3. Fill Proof 3 with actual SVD theory from mathlib4
4. Complete Proof 4: prove gyration is orthogonal transformation
5. Vectorize proofs: convert 1D scalar proofs to ℝ^n using `PiLp` norms
