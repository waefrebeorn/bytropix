# Vault: Lean Proofs — Formal Verification of WuBu Math
#
## Location: `MATH/lean/wubu_proofs/`

### 4 Proof Files
1. `PoincareBall.lean` — exp/log identity in Poincaré ball (partial proof)
2. `MobiusAdd.lean` — Möbius addition preserves the ball (1D proven)
3. `MLACompression.lean` — KV compression error bound (sketch)
4. `HyperbolicGyration.lean` — Gyration preserves the ball (sketch)

### mathlib4 Build
Currently compiling (lake build) — algebra, topology, ring theory modules.
Lean 4.29.1, lake 5.0.0.
s ~20 parallel workers.
    15|
    16|### Other Math Files
    17|- `MATH/wubu-formalism.md` — The central Wubu equation: Q = Σ q_k ∏ α_i^E
    18|- `MATH/README.md` — Math directory overview
    19|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
