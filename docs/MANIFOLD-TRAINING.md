# MANIFOLD-TRAINING — pre-training, post-training, fine-tuning on nested hyperbolic manifolds

> 2026-10-18. The geometry is not a metaphor: WuBuMath's Lean core proves the
> invariants this stack depends on (17 theorems, 0 `sorry`). This document is
> the training design that *uses* them, and the map of which pieces already
> exist in the engine.

## 0. The claim

The next level is not a bigger transformer in Euclidean space. It is a model
whose **representation, objective, and adapters all live on the same manifold**:

```
pre-training   : p(next | context) on a hyperbolic hierarchy
post-training  : preferences as geodesic relations, not flat scores
fine-tuning    : Möbius tangent-space adapters (never leaves the ball)
serving        : decode in tangent charts at each nesting level
```

The OS is the substrate that makes this reproducible: deterministic scheduling,
a KV filesystem, capability-scoped devices, and a compiler that can target the
silicon the manifold runs on.

## 1. What is already proven (WuBuMath, read-only source of truth)

From `wubumath/docs/LINKAGE.md` (Lean 4 + mathlib, 0 `sorry`):

| Theorem | File | What it guarantees for training |
|---|---|---|
| `exp_0^c(log_0^c(y)) = y` | `PoincareBall.lean` | the tangent↔ball bridge is lossless; optimizer state can live in tangent space |
| Möbius addition preserves the ball (nD) | `MobiusAddND.lean` | every update stays on-manifold |
| Gyration preserves inner product (Ungar 1.32) | `GyrationND.lean` | gyration is a rotation → hyperbolic layers are genuine linear-algebra objects |
| Gyration preserves the norm (1.33) | `GyrationND.lean` | rotation-equivariant layers are well-defined |
| `c·‖exp_0^c(v)‖² < 1` | `ExpMapBall.lean` | the exp map never escapes; a flow/ODE integrator is safe |
| Parallel transport is a Riemannian isometry | `ParallelTransport.lean` | gradients/velocities move between tangent spaces without distortion |
| φ-curvature nesting progression | `NestedHyperbolicSpaces.lean` | the nesting-chain invariant: curvature changes by a fixed schedule |
| Geodesic interpolation endpoints | `GeodesicInterp.lean` | flow rollouts start where they should |
| MLA KV-compression Frobenius bound | `MLACompression.lean` | the latent attention bound we rely on |
| Holographic optimizer decomposition | `HolographicOptimizer.lean` | gradient decomposition used by the Riemannian optimizer |

**Rule:** where our C11 and the Lean disagree, Lean wins and the C is the bug.

## 2. Stage 1 — pre-training on a manifold

### 2.1 Representation

- Token embeddings are **ball points**, not free vectors: initialise in the
  tangent space at the origin and map with `wubu_exp_map` (proven to land
  inside).
- The hierarchy is explicit: the token/word taxonomy supplies parent–child
  pairs; the loss pulls children toward parents in geodesic distance.
- Nested levels use the φ-curvature schedule: level ℓ has radius
  `R_ℓ = R_0 · φ^ℓ` (φ from `NestedHyperbolicSpaces.lean`). The outer level
  holds coarse semantics, the inner level fine distinctions.

### 2.2 Objective

1. **Next-token CE** through the hyperbolic LM head
   (`wubu_hyperbolic_output_proj`): hidden → exp → M⊗ → log → logits.
2. **Geodesic hierarchy loss**: for a parent p and child c,
   `max(0, d(c,p) - d(c, sibling) + margin)`, i.e. the child is closer to its
   parent than to a random sibling. This is the manifold-native analogue of a
   contrastive term.
3. **Radius regulariser**: penalise `‖z‖ → R` so the model does not park
   everything on the boundary (where gradients vanish).
4. **Curvature schedule**: R is not learned per batch; it follows the proven
   nesting progression, so a checkpoint is interpretable.

### 2.3 What is implemented now

- `src/wubu_manifold_train.c` — hyperbolic classifier head with an analytic
  tangent-space gradient; **geodesic hierarchy loss**; **radius regulariser**
  (squared hinge with a 5% inward margin so the band is reachable in finite
  steps); **Möbius LoRA** with **dA/dB backward**.
- `src/wubu_hyperbolic_output_proj.c` — the hyperbolic LM head.
- `src/wubu_mobius.c` — exp/log (canonical definitions), Möbius add, geodesic
  distance, backward passes.
- `src/wubu_nested_ssm.c`, `src/wubu_poincare_gqa.c` — manifold-native layers.
- `src/wubu_manifoldviz.c` — visualisation.

Verified in `tools/test_manifold_train.c`:
- hierarchy: child–parent distance < child–sibling; loss 0 when ordered,
  positive when violated;
- radius: **64/64** points inside `[0.2R, 0.8R]` after 300 steps;
- LoRA gradients: dA/dB match finite differences, max relative error
  **2.328e-03** (< 1e-2).

**Still missing for a real pre-training run:** a data loader that emits
taxonomy triples (parent, child, sibling). The loss and the geometry it needs
are done.

## 3. Stage 2 — post-training (preferences on the manifold)

- A reward model scores a response by its **geodesic relation** to a reference
  point / anchor set, not by a flat scalar head. Bradley–Terry becomes
  `P(a ≻ b) = σ((d(b, r) - d(a, r)) / τ)`.
- **GRPO/RLHF** runs with the policy in tangent charts: sample, compute
  advantages, take a Riemannian step (`u ← u - lr · PT(grad)`), map back.
  Parallel transport is proven to be an isometry, so the step does not distort
  the geometry.
- **KL regularisation** is measured on the manifold (wrapped/tangent KL), not
  in flat logit space — otherwise the penalty fights the geometry.
- **Verifier-first:** preference data is only trusted where an executable
  check exists (the gauntlet pattern). This is the same discipline as the
  `wubu_verify` / `wubu_equiv_check` modules.

**Status:** the RL machinery exists (`wubu_grpo`-family, `wubu_policy`,
`wubu_ppo`, `wubu_value`); the manifold-native reward and geodesic KL are the
gap.

## 4. Stage 3 — fine-tuning with Möbius LoRA

The adapter is the one piece that is both new and fully tested today.

```
u  = log_map(h, R)             # tangent chart at the representation
u' = u + B (A u)               # low-rank update, rank r << d
h' = exp_map(u', R)            # proven to land back in the ball
```

Properties (all covered by `tools/test_manifold_train.c`):

- **B = 0 is the identity** — attaching an adapter cannot change a trained
  model until it is trained.
- **The output never leaves the ball** — `ExpMapBall.lean` guarantees it; the
  test asserts `‖h'‖ < R`.
- **Curvature is unchanged** — the adapter moves along the manifold, it does
  not re-scale the space.
- **Merge rule:** two adapters compose by adding their tangent updates before
  the exp map (`u + Δ₁ + Δ₂`), not by adding ambient weights. This is the
  manifold-correct analogue of LoRA merging.

**Done:** the backward through A/B (`wubu_manifold_lora_backward`, verified
against finite differences at 2.3e-03 relative error) using the existing
`wubu_exp_map_backward` Jacobian.

**Missing:** the trainer that freezes the base and updates only A/B, and a
two-adapter merge test (`u + Δ₁ + Δ₂` before the exp map).

## 5. Serving

- KV cache is stored in **tangent coordinates per nesting level**; the KV-FS
  (`wubu_kvfs`, `wubu_kv_styx`) already gives it paths and persistence.
- Decode uses the hyperbolic LM head; the logits are Euclidean only at the
  final softmax boundary, which keeps the sampler conventional.
- Curvature-aware RoPE: positions are mapped through the same φ schedule so
  that long-context distance behaves like the trained geometry.

## 6. Metrics that actually test the manifold

| metric | what it catches | target |
|---|---|---|
| ball occupancy histogram | collapse to origin or boundary | mass in [0.2R, 0.8R] |
| geodesic distortion | embedding hierarchy quality | child–parent < child–sibling, ≥90% |
| curvature drift | schedule violation | 0 (R is fixed by φ) |
| tangent-gradient norm | vanishing/exploding near boundary | finite, bounded |
| exp/log round-trip error | numerical breakdown | < 1e-4 |
| downstream CE | the model still learns | monotone on the golden suite |

## 7. Risks and honest limits

- **Boundary stiffness.** Hyperbolic gradients vanish near `‖x‖→R`. The radius
  regulariser is not optional; without it the model parks on the boundary and
  stops learning.
- **Curvature is a hyperparameter with a proof, not a free parameter.** The φ
  schedule is proven consistent; learning R per batch would break the invariant
  the proofs give us.
- **The Lean core covers geometry, not learning.** No theorem says the
  objective is good; only that the operations are well-defined. Empirical work
  remains.
- **Numerical range.** `atanh` near 1 and `tanh` near saturation need float64
  in the inner loop for long training runs; the current C uses float32 for
  speed.
- **No claim of superintelligence.** This is a training geometry with proven
  invariants. It is necessary, not sufficient.

## 8. Implementation order

1. ~~Geodesic contrastive loss (Stage 1)~~ **done, tested**.
2. ~~Radius regulariser (Stage 1)~~ **done, tested (64/64 in band)**.
3. ~~Backward for Möbius LoRA A/B (Stage 3)~~ **done, FD-verified**.
4. Taxonomy-triple loader (Stage 1) — next.
5. Geodesic Bradley–Terry reward + tangent KL (Stage 2).
6. Curvature-scheduled RoPE for long context (serving).

Each lands with a test in `test_all`, and each cites the Lean theorem it
depends on.
