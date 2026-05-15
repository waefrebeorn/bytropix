# Vault: Theory — WuBu Nesting Physics & Philosophy

## Files (Physical Locations)

| File | Location | Format |
|------|----------|--------|
| WuBu Nesting (full paper) | `../../THEORY/WuBu_Nesting.md` | Markdown (27K chars, converted from PDF) |
| WuBu Nesting (PDF original) | `../../THEORY/WuBu_Nesting.pdf` | PDF (444KB) |
| WuBu Nesting v0.1 | `../../ENCODERS/hash-mind/WuBuNestingv0.1.md` | Markdown (36K chars, earlier draft) |
| Mind Palace condensed summary | `../mind-palace/tier1-core/1-wubu-theory/README.md` | Markdown (79 lines, key formulas) |
| Lean proofs | `../../MATH/lean/wubu_proofs/` | Lean 4 (4 proof files) |
| K-theory + Hopf | `../mind-palace/tier1-core/1-wubu-theory/README.md` §5-6 | Markdown |

## Core Philosophy
Standard AI learns by brute force — flattening globes onto paper. WuBu builds *inside the correct geometry*. Nested hyperbolic spaces (Russian doll) with learnable curvature/scale. Quaternion SO(4) rotations between levels. BSP trees for log-time attention. Hamilton encoders for quaternion compression.

## Key Equation
`Q = Σ q_k ∏ α_i^E` — the central WuBu formalism (see `../../MATH/wubu-formalism.md`)

## What We Actually Use in Code

| Operation | Formula | Where Used | Status |
|-----------|---------|------------|--------|
| exp_map | `tanh(\|x\|/R) × x/\|x\|` | Embedding → Poincaré (pre-forward) | ✅ Done |
| log_map | `R × artanh(\|x\|) × x/\|x\|` | Poincaré → LM Head (pre-output) | ✅ Done |
| Möbius add | `x⊕y` standard formula | SSM recurrence (Phase 2) | ✅ Done |
| Poincaré distance | `arcosh(1 + 2\|x-y\|²/((1-\|x\|²)(1-\|y\|²)))` | MoE router (Phase 4) | ✅ Done |
| RSGD optimizer | step in tangent space, project back | Training loop (Phase 3) | ✅ Done |
| Gyration closed-form | exact algebraic | SSM momentum correction | ✅ Done (3× faster) |

## Implementation Status (May 15 v6)
All hyperbolic operations now fully implemented in C/CUDA with verified backward passes.
Cold gaps closed: exp_map/log_map → Möbius → gyration → Poincaré distance → RSGD → output projection.

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) for navigation.*
