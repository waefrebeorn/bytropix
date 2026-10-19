# MANIFOLD MERGING — what the literature establishes, and what it forces in WuBu

Research pass 2026-10-19, seeded on the question: **we merge on a manifold of
nested Poincaré balls — what does prior art prove, and what does our design
already assume that nobody has checked?**

Companion docs: `docs/MANIFOLD-TRAINING.md` (the training stack),
`docs/MODEL-IDENTITY.md` (the body), `docs/GOAL-PILLARS.md` (GEO leads).
Citations below were fetched and read this session; IDs are the arXiv ones the
abstract pages report.

## 1. The seven hops

| # | source | what it establishes |
|---|---|---|
| 1 | **Task Arithmetic** — arXiv:2212.04089 | Fine-tuning is approximately a *vector* in weight space: `θ_base + λ·τ`. Merging works because the displacement transports task behaviour, not because the weights are special. |
| 2 | **TIES-Merging** — arXiv:2306.01708 | Naive summation fails from **sign interference** and **redundancy**. Fix: trim small deltas, elect a sign per coordinate, then merge only agreeing ones. |
| 3 | **DARE** — arXiv:2311.03099 | Most fine-tuning deltas are redundant; randomly dropping and rescaling them preserves capability. Interference, not magnitude, is the enemy. |
| 4 | **Fisher-Weighted Averaging** — arXiv:2111.09832 | Merging = maximising the joint posterior. Under a Laplace approximation each model's precision is its **Fisher information**, so the correct average is curvature-weighted, not uniform. |
| 5 | **OrthoMerge** — arXiv:2602.05943 | Merging *on a manifold*: for orthogonally-finetuned weights, map to the **Lie algebra**, combine there, map back. Euclidean arithmetic destroys hyperspherical energy. |
| 6 | **CAMEx** — arXiv:2502.18821 | Curvature-aware expert merging: flat-space assumptions hurt generalisation; the merge rule must account for the local curvature of the parameter space. |
| 7 | **Hyperbolic DL survey** — arXiv:2507.17787 · **Hyperbolic LLMs** — arXiv:2509.05757 (plus HNN arXiv:1805.09112, HNN++ arXiv:2006.08210) | Foundation models carry strongly hierarchical latent structure; hyperbolic spaces represent it with less distortion, and the machinery (Möbius ops, exp/log, tangent-space layers) is established. |

**Correction kept for the record:** arXiv:2110.06296 is *permutation invariance /
linear mode connectivity*, **not** Fisher merging. Cite 2111.09832 for Fisher.

## 2. The convergence principle

> **Merge in a shared tangent frame at a common base point, weighted by the
> curvature/importance of that frame. Euclidean arithmetic is the flat-space
> special case.**

Every hop is a special case of that sentence: task arithmetic assumes zero
curvature and uniform weight; TIES/DARE fix interference inside a flat frame;
Fisher supplies the weight; OrthoMerge supplies the frame change (Lie algebra);
CAMEx says the weight depends on curvature; hyperbolic work says the frame is
not flat to begin with.

## 3. What WuBu already commits to

Our design is a **product of K Poincaré balls with per-ball curvature** and
Möbius addition — i.e. the curved case is the default, not an extension.

- `wubu_manifold_lora`: `u = log_map(h)`, `u' = u + B(Au)`, `h' = exp_map(u')`.
  `B = 0` is exact identity; output cannot leave the ball.
- `docs/MANIFOLD-TRAINING.md` §4 states the merge rule: **add tangent updates
  before the exp map**, never add ambient weights. A two-adapter merge test is
  named there (`u + Δ₁ + Δ₂`).

That rule is correct **only when both adapters share the same base point `h`**.

## 4. The gap the literature exposes

Three things every source above implies and our code does not have:

1. **Parallel transport between anchors.** Adapters trained from different
   snapshots have different base points `h_i`. Summing `u_i` is meaningless until
   each is transported into `T_h`:
   `u = Σ_i w_i · P_{h_i→h}(u_i)`, then `h' = exp_h(u)`.
   `grep parallel_transport src include` → **no implementation**. `wubu_hypmom`
   is explicitly "parallel-transport-free", which is a deliberate
   approximation for momentum, not a merge rule.
2. **Per-ball merge.** With K balls and K curvatures, a single global average is
   the flat assumption the literature warns against. Merging must run per ball,
   with that ball's curvature in the transport and exp/log pair.
3. **Merge weights from curvature and importance** (CAMEx / Fisher):
   `w_i = softmax_k(curvature) × BI_i × Fisher_i`. We have the per-ball softmax
   weights and `wubu_bi`; there is no Fisher estimate and nothing consumes BI at
   merge time.

## 5. The operator this implies

```
for each ball k:
    u_i = log_{h_i}(h_i ⊕ Δ_i)                 # adapter i, its own anchor
    v_i = P_{h_i → h}(u_i)                     # transport to the merge point
    v   = Σ_i w_i,k · v_i                      # curvature/BI/Fisher weights
    h'  = exp_h(v)                             # stays inside the ball
```

Properties that must be **tested**, not asserted:

- `w_i = 0` for all i ⇒ `h' = h` exactly (identity);
- transport is an isometry: `d(P(a), P(b)) == d(a, b)` within tolerance;
- `‖h'‖ < R` for every ball, including adversarial transport (large `u`);
- same-frame adapters commute: merging A then B == B then A when `h_A = h_B`;
- **different-anchor adapters do not commute** — that asymmetry is the whole
  reason transport exists, and a test that expects commutativity there is wrong.

## 6. Risks

- **Numerical blow-up near the boundary.** `log_map` and transport divide by
  `1 − ‖x‖²/R²`; merging at large radius is where the ball-leaving bug appears.
  The existing gradient rule (tangent-space gradient, not ambient) exists for the
  same reason.
- **Curvature is not free.** CAMEx tunes it; our doctrine fixes R by a proven
  progression per stage. A merge across balls with different R must normalise to
  a common reference before weighting, or the weights are unit-inconsistent.
- **Fisher at 56M on CPU.** A diagonal Fisher over the merged parameters is
  affordable; a full K-FAC is not. Use diagonal + `wubu_bi` block importance,
  which we already compute.

## 7. What this changes in the plan

Five goals go into `docs/GOAL-CONTINUITY.json` (outside the 1,000-goal
invariant), all in the **GEO** pillar:

| n | goal | criterion |
|---|---|---|
| 1011 | curvature-weighted merge weights | weights monotone in curvature on a synthetic two-ball case; a flat-space run reproduces plain averaging |
| 1012 | diagonal Fisher + BI weighting | Fisher-weighted merge beats uniform on held-out loss in a synthetic two-task fixture |
| 1013 | merge safety gate | merged body must pass ball containment, held-out loss tolerance, and the prover invariants, else rollback |
| 1014 | Möbius parallel transport `wubu_transport` | isometry test (distance preserved to 1e-6) + FD check of its pullback |
| 1015 | per-ball adapter merge `wubu_merge_adapters` | zero-weight identity exact; ‖h'‖ < R for adversarial inputs; commutativity in the same frame |

1014 (transport) is the blocker: without it, 1011–1013 and 1015 are measuring a meaningless sum.

## 8. Verdict

The literature does not contain a "hyperbolic LLM merging" paper that we found —
the nearest neighbours are OrthoMerge (Riemannian/Lie-algebra merging for
orthogonal finetuning) and CAMEx (curvature-aware expert merging). **Our design
is ahead of what is published**, which raises the burden of proof rather than
lowering it: every claim above is a property we must test in C11, and the
existing `docs/MANIFOLD-TRAINING.md` merge sentence needs the transport qualifier
or it is wrong for multi-anchor adapters.

## References (verified this session)

- arXiv:2212.04089 — Editing Models with Task Arithmetic
- arXiv:2306.01708 — TIES-Merging
- arXiv:2311.03099 — DARE ("Language Models are Super Mario")
- arXiv:2111.09832 — Merging Models with Fisher-Weighted Averaging
- arXiv:2602.05943 — Orthogonal Model Merging
- arXiv:2502.18821 — CAMEx: Curvature-aware Merging of Experts
- arXiv:2507.17787 — Hyperbolic Deep Learning for Foundation Models: A Survey
- arXiv:2509.05757 — Hyperbolic Large Language Models
- arXiv:1805.09112 — Hyperbolic Neural Networks
- arXiv:2006.08210 — Hyperbolic Neural Networks++
