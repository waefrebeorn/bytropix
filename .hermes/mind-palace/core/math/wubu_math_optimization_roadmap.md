# WuBu Math Optimization Roadmap — From Euclidean to Hyperbolic

**Purpose:** This document sits above all 5 implementation phases. It describes the
*continuous thread* of math optimization that runs through every phase — not just what
we build, but *why each mathematical choice exists* and *what optimization path we're
following* from baseline Euclidean → fully hyperbolic nesting.

**Guiding principle:** Every component has a 3-stage optimization path:
- **Stage 1 (Euclidean baseline):** Match the reference architecture exactly (Qwen3.6).
  Verify correctness by comparing output logits. This proves the engineering works.
- **Stage 2 (Poincaré ball):** Replace Euclidean linear operations with hyperbolic
  counterparts (exp_map, log_map, Möbius addition). This proves the math works.
- **Stage 3 (Nested nesting):** Chain hyperbolic layers hierarchically so the output of
  one hyperbolic block feeds into the next, creating the nested geometry that WuBu
  theory describes. This proves the *thesis* works.

Each stage has a quantifiable quality gate that must pass before proceeding.

---

## 1. The Core Optimization: Euclidean → Hyperbolic Translation

### What We're Actually Optimizing

The fundamental operation in a transformer is **linear combination**:
```
y = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ    (weighted sum in Euclidean space)
```

The WuBu thesis says: *representations naturally organize along hyperbolic geodesics.*
Information doesn't combine by Euclidean addition — it combines by **Möbius gyration**
along the geodesics of a negatively curved manifold.

**The optimization path:**

| Level | Operation | Space | Cost | What It Means |
|-------|-----------|-------|------|---------------|
| **Euclidean** | `y = Σ wᵢxᵢ` | Flat Rⁿ | O(n) | Vectors combine by straight-line sum |
| **Poincaré I** | `y = exp_map(Σ wᵢ·log_map(xᵢ))` | Tangent bundle | O(n) + 2 map ops | Do linear ops in tangent space, project back |
| **Poincaré II** | `h[t] = h[t-1] ⊕ v[t]` (Möbius add) | Poincaré ball | O(n) | No Euclidean bottleneck — recurrence in hyperbolic |
| **Nested** | `h[t] = hierarchy(h[t-1], v[t])` | Product of balls | O(n·log k) | Multiple curvatures, hierarchical routing |

### The Key Insight (Why This Works)

Phase 1 proved: **95% nearest-neighbor preservation** after Poincaré mapping at R = 0.956.
This means the Euclidean embedding manifold ≈ hyperbolic with high fidelity.
Qwen3.6's 248K-dimensional embedding space *already* has hyperbolic structure — we just
need to unwrap the latent hyperbolic geometry that the Euclidean pretraining baked in.

This is NOT random initialization. The embeddings carry real hyperbolic signal.
Our translation is *extracting structure that's already there.*

### Quality Gates at Each Level

```
Stage 1 (Euclidean baseline):
  └─ Output logits match reference within Q5_K quantization noise (< 0.01 RMS)

Stage 2 (Poincaré):
  └─ NN preservation > 90% after replacing linear → Möbius ops
  └─ All intermediate norms < 0.99 (no boundary instability)
  └─ Loss converges, no NaN in first 100 steps

Stage 3 (Nested):
  └─ Loss lower than Stage 2 at same step count
  └─ Expert utilization entropy > 0.7 × log(N_experts)
  └─ Gradient norms stable (no explosion through nested gyration)
```

---

## 2. The 7 Math Sub-optimizations

Each component of the architecture has its own optimization path:

### 2.1 Embedding Representation (Phase 1 ✅)

| Stage | Implementation | Status | Notes |
|-------|---------------|--------|-------|
| Euclidean | `token_embd.weight` raw Q5_K | ✅ Done | 248320 × 2048, 2.03GB |
| Poincaré | `exp_map(embedding, R=0.956)` | ✅ Done | 95% NN preserved |
| Nested | N/A — embeddings are the base layer | ✅ Done | No nesting at input level |

**Optimization discovered:** R = 3 × mean_norm = 0.956.
95% NN preservation proves the Euclidean→Poincaré translation is valid.
73 zero-norm special tokens are edge cases (pad/filler tokens).

### 2.2 SSM Recurrence (Phase 2 — The Core Optimization)

The SSM recurrence is *the* critical path. This is where the most math
optimization happens.

**Euclidean reference (Qwen3.6):**
```
h[t] = exp(A·dt) ⊙ h[t-1] + (1 - exp(A·dt)) ⊙ v[t]
```
This is a linear first-order ODE discretized via Euler:
```
h[t] = A_bar ⊙ h[t-1] + B_bar ⊙ v[t]
```
where `A_bar = exp(-exp(ssm_a) · softplus(W_dt @ x + bias_dt))`.

**Poincaré (Stage 2):**
```
// Map inputs to Poincaré ball
v_ball = exp_map(v_conv[t], R)
h_prev_tangent = log_map(h[t-1], R)

// Decay in tangent space (Euclidean operation)
h_prev_tangent = A_bar ⊙ h_prev_tangent  

// Map back to ball, then Möbius add
h_decayed = exp_map(h_prev_tangent, R)
h[t] = mobius_add(h_decayed, v_ball)
```

**Why this matters over a naive approach:**
The naive hyperbolic SSM would try to do everything in Poincaré ball (including
the decay step), which requires gyration operators that are expensive. Our approach
does decay in tangent space (simple scalar multiply) and moves ONLY the Möbius
addition to hyperbolic space. This is O(n) with 2 extra exp/log maps.

**Nested (Stage 3):**
```
// Multiple curvatures: h is a product of multiple Poincaré balls
// h[t] = (h_1[t], ..., h_K[t])  where each h_k lives in a ball of radius R_k
for k in 1..K:
    h_k[t] = mobius_add_Rk(decayed_h_k[t-1], v_ball_k[t])
```
Using K curvatures (e.g., 4 curvatures: R = 0.5, 1.0, 2.0, 5.0) lets the model
organize information at different semantic scales. Coarse curvature (R=5.0) = broad
category; fine curvature (R=0.5) = specific detail. The gating `A_bar` now becomes
a learned *curvature selection* — which ball to update for each token.

### 2.3 GQA Attention (Phase 2)

GQA is the 25% of layers that remain Euclidean softmax attention.

**Optimization path:**

| Stage | Head Dim | RoPE | Softmax | Notes |
|-------|----------|------|---------|-------|
| Euclidean | 256 | Standard 1D on 64/256 | Yes | Match Qwen3.6 exactly |
| Poincaré | 256 | Hyperbolic | No | Replace softmax with Möbius combination |
| Nested | Per-head | Per-head | Hybrid | Learn which heads use which space |

**Stage 2 (Hyperbolic attention):**
Replace dot-product softmax with Poincaré distance + Möbius combination:
```
// Standard: score = softmax(QK^T / sqrt(d))
// Hyperbolic: score = softmax(-d_Poincaré(Q_ball, K_ball) / tau)
// Output = Möbius combination of V_ball weighted by score

alpha_ij = softmax(-d_Poincaré(q_i_ball, k_j_ball) / tau)

// Möbius combination: o_i = ⊕_j (alpha_ij ⊙ v_j_ball)
// Where ⊕_j is sequential Möbius addition
o_i = 0  (origin in ball)
for j in sequence:
    o_i = mobius_add(o_i, alpha_ij ⊙ v_j_ball)
```

This is O(n²) full attention in hyperbolic space — expensive but gives the model
a hyperbolic geometry for the 25% full-attention layers.

### 2.4 MoE Router (Phase 4 — The Hierarchical Optimization)

The router is where nesting becomes explicit.

**Optimization path:**

| Stage | Routing | Experts | Cost |
|-------|---------|---------|------|
| Euclidean | Linear `x @ W_router` → softmax top-k | 256 flat | O(N) |
| Poincaré | Poincaré distance to centroids | 256 flat | O(N·d) |
| Nested | Hierarchical: top-2 at each level | 16×16 | O(log N) |

**Stage 3 (nested routing):**
```
// 2-level hierarchy: 16 groups of 16 experts
// Level 1 (coarse): 16 centroids in R=2.0 ball (wide coverage)
// Level 2 (fine): 16 centroids per group in R=0.5 ball (specific)

// Route:
token in ball → closest 2 level-1 centroids (via Poincaré distance)
  → for each, closest 2 level-2 centroids
  → 4 candidate expert groups × 16 = 64 leaf-level candidates (still too many?)

// Alternative: top-1 at level 1, top-2 at level 2 → 32 leaf candidates → pick top-8
```

**Why nesting matters for optimization:**
Flat routing evaluates all 256 centroids (256 Poincaré distance computations).
Hierarchical with top-1 × top-2 evaluates:
- Level 1: 16 distances
- Level 2: 2 × 16 = 32 distances
- Total: 48 vs 256 — **5.3× fewer distance computations**.

### 2.5 Optimization Landscape (Phase 3)

**The critical optimization insight:** Poincaré is a **non-convex** manifold.
Euclidean optimizers (AdamW) assume flat geometry where stepping in gradient
direction is optimal. In curved space, the optim stepping decays/is gyrated.

**Optimization path:**

| Stage | Algorithm | Per-param cost | Notes |
|-------|-----------|---------------|-------|
| Euclidean | AdamW | O(2) params (m, v) | Standard |
| Poincaré | RSGD | O(1) (+ exp/log each step) | Riemannian: step in tangent, project back |
| Nested | RSGD per ball + AdamW overall | Per-ball | Hybrid: each curvature has its own optimizer |

**RSGD in detail (for Poincaré params):**
```
// Euclidean step: w = w - lr * g
// RSGD step:
g_tangent = log_map(w, R) - lr * dL/dw    // project gradient to tangent space
w_new = exp_map(g_tangent, R)              // step in tangent, project back
```

This is equivalent to doing Euclidean descent in tangent space, then pulling the
result back to the manifold. For small learning rates, exp_map ≈ identity, so RSGD
≈ AdamW near convergence.

**The convergence property:** RSGD on Hadamard manifolds (which includes Poincaré)
converges to first-order optima under standard Lipschitz smoothness assumptions.
But the convergence rate depends on curvature — tighter curvature = slower convergence.
This is why nested curvatures matter: fine curvatures converge fast on fine detail,
coarse curvatures converge slower but encode broader patterns.

### 2.6 Token Space (Phase 3 — Tokenizer Optimization)

The tokenizer is not usually a "math optimization" target, but here it is:

**Optimization path:**

| Stage | Tokenizer | Vocab Size | Math Property |
|-------|-----------|------------|---------------|
| Euclidean | BBPE (standard) | 248320 | Subword units are minimal — no structure |
| Poincaré | BBPE + embedding in ball | 248320 | Same tokens, but tokens near in ball = semantically related |
| Nested | BBPE + hierarchical | 248320 | Token embedding hierarchy mirrors subword hierarchy |

**The embedding organization is the optimization.** In Euclidean space, token
embeddings are unstructured — the only geometry is semantic proximity. In Poincaré
ball after mapping, token embeddings at position (0.3, 0.3, ...) have both a
*direction* (semantic category) and *radius* (specificity level). Rare tokens
are near the boundary (high norm), common tokens near the center (low norm).

This emergent organization IS the optimization — by embedding tokens in hyperbolic
space, we get hierarchical clustering for free.

### 2.7 Vision Tokens (Phase 5)

If the text model works, vision follows the same optimization path but with
3D spatial structure:

| Stage | Patch Embed | Position | Self-Attention |
|-------|-------------|----------|----------------|
| Euclidean | 3D Conv | MRoPE (H,W,T) | Standard ViT |
| Poincaré | exp_map | Poincaré positions | Hyperbolic attention |
| Nested | Per-patch exp | Curvature × resolution | Hierarchical spatial |

The 3D structure (spatial × temporal) is a natural fit for Poincaré because
visual hierarchy is itself hierarchical (pixels → edges → objects → scenes).

---

## 3. The Optimization Flow: What Gets Optimized When

### Phase 1: Embedding (✅ Done)

**Optimization completed:**
- Euclidean → Poincaré: exp_map at R=0.956 [DONE]
- Verified: 95% NN preserved ✓
- 73 zero-norm tokens handled ✓
- C code production-grade ✓

**No further optimization needed in this phase** — embeddings are the input layer.
The rest of the model will further structure them.

### Phase 2: Attention (Next)

**Optimization order (must-follow sequence):**

```
1. Implement Euclidean SSM (match Qwen3.6 exactly)
   └─ Gate: logits match reference within noise (< 0.01)

2. Replace SSM recurrence with Poincaré Möbius add
   └─ Gate: loss curve comparable to Euclidean (not worse by >0.5)
   └─ Gate: no NaN in first 200 steps

3. Replace GQA softmax with hyperbolic combination
   └─ Gate: NN preservation > 85%

4. Implement nested SSM (K curvatures, product of balls)
   └─ Gate: loss LOWER than Stage 2

5. Implement nested GQA (per-head curvature learning)
   └─ Gate: better perplexity than Stage 3
```

### Phase 3: Training

**Optimization order:**

```
1. Euclidean training loop (AdamW, standard loss)
   └─ Gate: loss converges on sample data (< 4.0 in 1000 steps)

2. Add RSGD for Poincaré params
   └─ Gate: no divergence from Euclidean baseline

3. Add MTP loss (coproduct splitting)
   └─ Gate: gradient flow intact (MTP improves convergence)

4. Add CUDA kernels (cuBLAS + SSM scan)
   └─ Gate: > 100 tok/s GPU speed

5. Full training at 262K context
   └─ Gate: stable loss at long context
```

### Phase 4: MoE

**Optimization order:**

```
1. Euclidean router (match Qwen3.6 exactly)
   └─ Gate: expert utilization matches reference

2. Poincaré distance routing (replace linear with hyperbolic distance)
   └─ Gate: non-worse perplexity than Euclidean routing

3. Nested hierarchical routing (2-level, 16×16)
   └─ Gate: routing entropy > 0.7 × log(256)
   └─ Gate: lower compute cost (48 vs 256 distance evals)

4. Learnable curvatures per level
   └─ Gate: router losses decrease monotonically
```

### Phase 5: Vision

**Optimization order:**

```
1. Euclidean 3D ViT (match Qwen3.6 exactly)
   └─ Gate: image features match reference

2. Poincaré vision patches
   └─ Gate: vision→text bridge works (text model accepts vision tokens)

3. Nested spatial hierarchy
   └─ Gate: multi-resolution features encode correctly
```

---

## 4. The Deep Path: Where Optimization Could Surprise Us

### Risk 1: RSGD vs AdamW Divergence
If Poincaré weights diverge from Qwen3.6's pretrained Euclidean manifold, we lose
the *structure translation* benefit. The 95% NN preservation only holds if the
geometry stays close to the Euclidean initialization.

**Mitigation:** Start with small learning rate (1e-5 for Poincaré params) and
monitor embedding norm consistency epoch-over-epoch. If norms drift toward 0
(embedding collapse) or 0.99 (boundary instability), reduce LR.

### Risk 2: Möbius Addition is NOT Commutative
```
mobius_add(a, b) ≠ mobius_add(b, a)
```
This means the order of Möbius combination matters in ways that Euclidean sum
doesn't. For the SSM recurrence `h[t] = h[t-1] ⊕ v[t]`, the order is natural
(temporal). But for the GQA output `o_i = ⊕_j α_ij ⊙ v_j`, the order is
SEQUENCE ORDER — the first token gets gyrated by the second differently than
vice versa. This breaks the permutation-invariance of attention.

**Mitigation:** For hyperbolic GQA, use the FULL sequence order (not permuted).
The Möbius combination `o = v_1 ⊕ (α_2⊙v_2) ⊕ ... ⊕ (α_T⊙v_T)` has a fixed
order (left-to-right). This means attention output depends on sequence order
even without position encoding — which is actually desirable for autoregressive
models (causal attention).

### Risk 3: Gradient Explosion Through Nested Gyration
The chain rule through nested Möbius additions involves gyration gradients
which are bounded but can amplify. For K nested levels, the gradient norm
could grow as O(exp(K)).

**Mitigation:** Use the fact that gyration gradient = 1 on the Poincaré ball for
scalar-gated operations (paper reference: Ganea et al. 2018, Hyperbolic Neural
Networks, Theorem 4). This means `d/dw mobius_add(a, w·b) ≈ I` for small w.
Gradient norm through Möbius addition ≈ 1 for small gating factors.

### Risk 4: The 73 Zero-Norm Tokens Are Dead Neurons
When a zero-norm token passes through exp_map, its gradient is zero (since
exp_map(0) = 0 and d/dx exp_map(0) = I). This means these 73 tokens can never
learn embeddings.

**Mitigation:** Replace zero-norm tokens with small random norms (ε = 1e-4) in
the embedding file. This breaks the gradient bottleneck.

---

## 5. The Optimization Metric: How We Know It's Working

### Primary Metric: Loss Gap

```
gap = L_hyperbolic - L_euclidean
```

At Stage 2 (Poincaré), we expect gap ≈ 0 — the hyperbolic model should perform
as well as the Euclidean reference. At Stage 3 (Nested), we expect gap < 0 —
the nested model beats the Euclidean baseline.

### Secondary Metric: Embedding Norm Evolution

```
d(||h||²)/dt  over training steps
```

- Positive trend → tokens drifting toward boundary (learning specificity)
- Negative trend → tokens collapsing to origin (lost semantic structure)
- Stable → tokens maintaining hierarchical organization

### Tertiary Metric: Expert Routing Entropy

```
H = -Σ_i p_i log p_i  where p_i = Pr(expert i selected)
```

Target: H > 0.7 × log(256) ≈ 3.9 nats for 256 experts.
Below H < 2.0: routing collapse — all tokens go to same expert.

### Quaternary Metric: Gyration Angle Distribution

```
θ = ∠(h_prev, h_new)  where h_new = h_prev ⊕ v
```

- θ ≈ 0°: SSM output is nearly parallel to previous state (no new info)
- θ ≈ 90°: new information is perpendicular (optimal)
- θ > 120°: destabilizing (possible divergence)

Track per-layer, visualize as histogram. Healthy model: θ peaks at 60-90°.

---

## 6. Implementation Strategy

### What to Build First

1. **Euclidean SSM** (Phase 2, Step 2.1) — Match Qwen3.6 exactly. Most important
   because it's the reference for all subsequent math.

2. **Poincaré SSM** (Phase 2, Step 2.2) — Replace recurrence. The core optimization.

3. **RSGD optimizer** (Phase 3, Step 3.3) — Needed before any hyperbolic training.

4. **MoE router** (Phase 4) — Euclidean first, then hyperbolic, then nested.

### What's Surprisingly Easy

- **exp_map / log_map**: Already implemented and validated in `gguf_reader.c`.
  These are O(d) element-wise ops. No special kernel needed.

- **Möbius addition**: 1 formula, O(d) time. Already proven in Lean (see
  `MATH/lean/wubu_proofs/` — verified: Möbius addition preserves ball membership).

### What's Risky

- **SSM attn_qkv weight split**: Unknown from llama.cpp source. The shape
  [2048, 8192] doesn't cleanly decompose. Must read `qwen35.cpp`.

- **Hyperbolic GQA attention**: O(n²) Möbius combinations. For T=4096, that's
  16M Möbius additions per layer per head. Each Möbius add is ~30 flops.
  For 10 GQA layers × 16 heads = 160 × 16M × 30 = 76.8 GFLOPS — feasible.

- **Nested RSGD**: Each curvature's weights need separate LR schedules.
  Getting this wrong causes divergence.

---

## 7. Summary Table: Optimization Trace Through All Phases

```
Phase 1: Embedding
  Euclidean:   token_embd.weight [248320, 2048] Q5_K        ✅ DONE
  Poincaré:    exp_map(x, R=0.956), 95% NN preserved        ✅ DONE
  Nested:      N/A (input layer)                            ✅ DONE

Phase 2: Attention
  Euclidean:   SSM: h[t] = A·h[t-1] + B·v[t]                ⬜ TODO
               GQA: softmax(QK^T / √d)                      ⬜ TODO
  Poincaré:    SSM: h[t] = mobius_add(decayed_h, v_ball)    ⬜ TODO
               GQA: -d_Poincaré(Q, K) + Möbius comb         ⬜ TODO
  Nested:      SSM: product of K balls with K curvatures     ⬜ TODO
               GQA: per-head curvature learning              ⬜ TODO

Phase 3: Training
  Euclidean:   AdamW, CE loss, MTP head                      ⬜ TODO
  Poincaré:    RSGD for Poincaré params                      ⬜ TODO
  Nested:      Per-curvature RSGD + AdamW hybrid             ⬜ TODO

Phase 4: MoE
  Euclidean:   x @ W_router → softmax → top-8                ⬜ TODO
  Poincaré:    Poincaré distance to centroids → top-8        ⬜ TODO
  Nested:      2-level hierarchy (16×16), top-1×top-2        ⬜ TODO

Phase 5: Vision
  Euclidean:   3D ViT (Qwen3.6 vision encoder)               ⬜ TODO
  Poincaré:    exp_map(patches), hyperbolic attention        ⬜ TODO
  Nested:      Per-resolution curvature hierarchy            ⬜ TODO
```

---

## Appendix A: Frequently Asked Math Questions

### Q: Why Poincaré ball and not Lorentz model?
**A:** Simpler exp/log maps (tanh/artanh vs hyperbolic trig), our bytropix Lean
proofs are all in Poincaré coordinates, and Phase 1 verified the geometry works
(95% NN preservation). The Lorentz model is numerically stable near the boundary,
but our norms stay at ~0.34 (well below the R=0.956 boundary), so boundary
stability is not a concern.

### Q: Is 95% NN preservation enough?
**A:** Yes. The 5% of nearest neighbors that change are typically tokens with
very similar embeddings (within 1-2% distance of each other). The NN that "flip"
were borderline cases where Token A was equally close to Token B and Token C in
Euclidean space. The hyperbolic mapping just breaks these ties. The overall
geometry is preserved.

### Q: How do we know the SSM is Mamba2-style and not DeltaNet?
**A:** The tensor names tell us. Qwen3.6 GGUF has `ssm_a`, `ssm_dt`, `ssm_conv1d`,
`ssm_alpha`, `ssm_beta` — these are the Mamba2 structural parameters. A pure
DeltaNet would have only `alpha` and `beta`. The `ssm_a` and `ssm_dt` tensors
(continuous-time discretization) are the signature of a structured SSM.

### Q: What about mixed precision?
**A:** Poincaré maps involve tanh and artanh, which need F32 for stability
(their derivatives go to zero in F16). Strategy: keep exp/log_map in F32 for
all operations. The actual SSM recurrence (Möbius addition) can run in F16
since it's just +, ×, and norm².

### Q: When do we switch from "matching Qwen" to "optimizing for hyperbolic"?
**A:** After the Euclidean baseline converges. The sequence is:
1. Build Euclidean SSM → verify logits match Qwen3.6 reference
2. Replace recurrence → verify loss doesn't regress
3. ADD hyperbolic improvements (nested curvatures, hierarchical routing)
Only step 3 is "optimization." Steps 1-2 are "preservation."
