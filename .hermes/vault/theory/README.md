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
| Poincaré distance | `arcosh(1 + 2\|x-y\|²/((1-\|x\|²)(1-\|y\|²)))` | MoE router (Phase 4) | 🔄 Planned |
| RSGD optimizer | step in tangent space, project back | Training loop (Phase 3) | 🔄 Planned |

## Update Note (May 13)
The vault theory directory previously listed files (01-foundational-philosophy.md, 02-axiomatic-emergent-theory.md, etc.) that did not exist as separate files — those sections are incorporated into `../../THEORY/WuBu_Nesting.md`. This README now links to the actual physical file locations.
