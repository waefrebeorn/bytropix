# Phase 4: MoE Port — WuBu Nested Geometry Routing

**Goal:** Replace Qwen3.5+ linear router with wubu hyperbolic nested routing.

**Depends on:** Phase 2 (attention + gyration), Phase 3 (training loop)

## Target MoE Spec (Qwen3.6-35B-A3B)

From GGUF metadata and tensor shapes:
```
num_experts: 256
num_experts_per_tok: 8 routed (+ 1 shared, always active)
expert_intermediate_size: 512
shared_expert_intermediate_size: 512
router_aux_loss_coef: 0.001
```

**Key shape difference from my earlier estimate:** The expert tensors have shape
`[2048, 512, 256]` not `[2048, 512]` — the 256 expert dimension is LAST, not first.
This means the weights are stored as `[input_dim, expert_dim, num_experts]` in column-major
GGUF format: for each expert (256), a matrix [2048, 512] for the gate projection.

So loading expert weights:
```
ffn_gate_exps[b, :, :] = weight[:, :, b]   // [2048, 512] for expert b
ffn_up_exps[b, :, :]   = weight[:, :, b]   // [2048, 512]
ffn_down_exps[b, :, :] = weight[:, :, b]   // [512, 2048]

// Same for IQ2_XS / IQ1_S quantized
```

**Shared expert is stored separately** (not as an expert index):
```
ffn_gate_shexp.weight  = [2048, 512]  Q8_K
ffn_up_shexp.weight    = [2048, 512]  Q8_K
ffn_down_shexp.weight  = [512, 2048]  Q8_K
```

**Router weight:** `ffn_gate_inp.weight = [2048, 256]` F32 — linear projection from
hidden to 256 expert scores. ALSO `ffn_gate_inp_shexp.weight = [2048]` F32 — a small
bias/projection for the shared expert routing (single scalar per hidden dim).

## Step 4.1: Standard MoE Router (Reference — matches Qwen3.5)

```c
// Standard: linear router from Qwen3.5
// Input: x[B, T, 2048]
// Router weights: W_router[2048, 256]  (ffn_gate_inp.weight)

// Step 1: Compute routing scores
float* router_logits = x @ W_router;          // [B, T, 256]

// Step 2: Add shared expert bias
float* shared_logit = dot(x, W_shared_router); // [B, T]  (ffn_gate_inp_shexp)

// Step 3: Softmax over 256 experts + shared
float* routing_weights = softmax(router_logits, dim=-1);  // [B, T, 256]

// Step 4: Top-k selection  
int top_k_indices[B, T, 8];
float top_k_weights[B, T, 8];
top_k(routing_weights, 8, top_k_indices, top_k_weights);
// Re-normalize top-k weights to sum to 1
top_k_weights /= sum(top_k_weights);

// Step 5: Dispatch
float* y = zeros[B, T, 2048];

// Shared expert (always active)
y += shared_expert(x);  // uses ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp

// Routed experts (top-8)
for each token t in batch:
    for each expert e in top_8_for_token:
        h = silu(x[t] @ ffn_gate_exps[e]) * (x[t] @ ffn_up_exps[e])  // SwiGLU
        y[t] += weight[e] * (h @ ffn_down_exps[e])
```

**Critical detail:** Qwen3.5 MoE uses SwiGLU activation (silu × linear), not just
a single FFN. The gate and up projections are element-wise multiplied.

## Step 4.2: Extract Expert Embedding Centroids for WuBu Routing

Before we can replace the router, we need to understand what each expert "specializes in":

1. Run a forward pass through the Qwen3.5 model (via llama.cpp) on a diverse corpus
2. Record which experts fire for each token
3. Compute centroid of POINCARÉ-MAPPED hidden states that route to each expert
4. These centroids become the initial positions for hyperbolic routing

Without this step, we'd be routing in the dark. The centroids must be data-driven.

**Alternative:** Initialize centroids via K-means on the Poincaré-mapped embedding vectors
(248320 points in Poincaré ball, cluster into 256 groups). This is faster and doesn't
require a full model forward pass. The assumption is that experts specialize in different
token types, which correlates with embedding similarity.

## Step 4.3: WuBu Hyperbolic Router (Replacement)

Replace the linear `ffn_gate_inp.weight` with hyperbolic distance routing:

```c
// WuBu: hyperbolic distance routing
// Centroid for each expert: c_e in Poincaré ball (2048-dim)
float centroids[256][2048];  // from K-means on Poincaré embeddings

// Step 1: Map input to Poincaré ball
float x_ball[2048];
wubu_exp_map(x, 2048, R, x_ball);

// Step 2: Compute Poincaré distances to each centroid
float dist[256];
for (int e = 0; e < 256; e++) {
    // Poincaré distance: d(x,y) = arcosh(1 + 2*||x-y||²/((1-||x||²)*(1-||y||²)))
    float diff_sq = wubu_norm_sq_diff(x_ball, centroids[e], 2048);
    float x_norm_sq = wubu_norm_sq(x_ball, 2048);
    float c_norm_sq = wubu_norm_sq(centroids[e], 2048);
    
    // Clamp for numerical stability
    x_norm_sq = fminf(x_norm_sq, 0.99f);
    c_norm_sq = fminf(c_norm_sq, 0.99f);
    
    float poincare_arg = 1.0f + 2.0f * diff_sq / ((1.0f - x_norm_sq) * (1.0f - c_norm_sq) + 1e-8f);
    dist[e] = acoshf(poincare_arg);
}

// Step 3: Convert distances to routing weights (closer = higher score)
// Use negative distance + softmax
float scores[256];
for (int e = 0; e < 256; e++) {
    scores[e] = -dist[e] / TEMPERATURE;  // temperature for routing sharpness
}
// Softmax
float routing_weights[256];
softmax(scores, 256, routing_weights);
```

**Hybrid approach:** Don't fully replace the router initially. Instead, interpolate:

```
final_scores = α * linear_scores + (1-α) * hyperbolic_scores
```

Start with α=0.0 (pure hyperbolic), increase α if routing collapse detected.

## Step 4.4: Nested Geometry

Instead of flat 256 centroids, organize hierarchically with 3-bit quantization per level:

```
Level 0 (4 centroids):    4 = 2 bits
Level 1 (4 per L0):      16 = 4 bits
Level 2 (4 per L1):      64 = 6 bits
Level 3 (4 per L2):     256 = 8 bits (actual experts)
```

**Routing:** Walk the tree top-down, pick top-2 at each level:
```
// At each level:
// - Compute Poincaré distance to centroid[level][branch]
// - Pick 2 closest branches
// - Descend into those 2 branches for next level
// Total: 4×4×4×4 = 256 leaves reachable
// Cost: 4 + 8 + 16 + 32 = ~60 distance computations vs 256 for flat
```

The 4-level tree means tokens nav by semantic proximity — similar concepts route
through similar branches. This mirrors WuBu's nested geometry thesis.

## Step 4.5: Shared Expert

Keep as-is from Qwen3.5. The shared expert handles common knowledge that doesn't
need expert specialization. It uses a separate set of weights (`ffn_*_shexp`).

**Note:** The shared expert `ffn_gate_inp_shexp.weight` [2048] is a single vector
(not a matrix). This suggests the shared expert gate is a scalar per hidden dimension,
not a full projection. This is a learned "relevance score" for whether to amplify
or attenuate the shared expert contribution.

## Files to Create
```
src/wubu_moe.c                  — MoE routing + dispatch (standard)
include/wubu_moe.h              — Header
src/wubu_moe_hyperbolic.c       — Hyperbolic distance router
include/wubu_moe_hyperbolic.h   — Header
src/wubu_cluster_init.c         — K-means on Poincaré embeddings for centroid init
include/wubu_cluster_init.h     — Header
tools/compute_expert_centroids.py  — Python script for centroid precomputation
```

## Pitfalls

1. **Centroid initialization is critical.** Random centroids in Poincaré ball will cause
   routing collapse (all tokens go to nearest centroid, which is random).
   - Fix: Initialize with K-means on embedding vectors (248320 points → 256 clusters).
   - The Python script `compute_expert_centroids.py` does this offline.

2. **Routing collapse** — all tokens route to the same expert.
   - Fix: Keep aux loss (0.001). Add load balancing loss: encourage uniform expert usage.
   - Monitor: expert utilization entropy. Target: > 0.8 × log(256).

3. **IQ2_XS/IQ1_S dequantization overhead** — the expert weights are heavily quantized
   (2-bit and 1-bit). Dequantizing on-the-fly during the forward pass adds overhead.
   - Fix: Cache frequently-used experts in f16 (hot cache). Evict least-recently-used.
   - For training: keep active experts in f16, the rest in compressed form.

4. **MoE dispatch complexity** — 256 experts × 8 active per token means we must
   gather-scatter efficiently. Naive implementation loops over 8 experts per token,
   which is 8 × B × T separate matmuls.
   - Fix: Group tokens by expert assignment, do one batched matmul per expert.
   - This requires a permutation step which is O(B × T × log(256)) for the sort.

5. **Nested tree vs flat routing** — the 4-level tree approximation may be worse than
   flat routing because errors propagate down the tree (a wrong choice at level 0
   means the token can never reach the correct level-3 expert).
   - Fix: Use a 2-level hierarchy (16 groups of 16) instead of 4-level. Simpler,
     fewer propagation errors, still O(√N) instead of O(N).
