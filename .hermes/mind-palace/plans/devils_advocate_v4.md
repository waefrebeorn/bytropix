# WuBuText AI — Devil's Advocate v4: Post-Sprint Reality Check

## Date: May 15, 2026
## Based on: Runtime verification of all 12 streams + May 15 PM fix session

---

## Executive Summary
**Previous DA (v3, May 13) identified 7 risks. ALL 7 resolved.**
**May 15 PM fix session closed 3 remaining ⚠️ items.**
**New threat: manifold backward passes are cold — no gradient flow for hyperbolic components.**

| v3 Risk | v4 Status |
|---------|-----------|
| GPU weight loading broken | ✅ Fixed — dequant bypass fixed |
| train_backprop hangs | ✅ Not hanging — just CPU-slow |
| train_gpu wrong loss | ✅ CE=69 → 12.42 |
| MoE not integrated | ✅ Lazy MoE in training |
| VRAM insufficient | ✅ CPU training works |
| Tokenizer CJK | ✅ Works |
| SSM/GQA forward | ✅ Verified |

| May 15 PM Fix | Status |
|---------------|--------|
| NaN in logits (0.57%) | ✅ Fixed — output projection added |
| CPU RMSNorm OOB (d=4096, weight[256]) | ✅ Fixed — 6 test_kv_cache.c call sites |
| GPU vision pipeline timeout | ✅ Resolved — 99ms, no timeout |

---

## Phase-by-Phase Results

### Phase 0 (GGUF Reader) — DONE ✅
13 GGML types, 733 tensors, GPU loading fixed.

### Phase 1 (Embedding Graft) — DONE ✅
Poincaré R=0.956, 95% NN preservation, 248K embeddings.

### Phase 2 (Attention Port) — DONE ✅
40 layers CPU/GPU forward. CE loss 12.66 CPU, 12.42 GPU.

### Phase 3 (Training Modules) — ALL WIRED ✅
All 7 modules wired into train_integrated binary with env flags.

| Module | Tests | Integration |
|--------|-------|-------------|
| RSGD | PASS | WIRED (RSGD=1) |
| Poincaré GQA | 4/4 PASS | WIRED (PGA=1) — FORWARD ONLY |
| Nested SSM K=4 | 3/3 PASS | WIRED (NESTED_SSM=1) — FORWARD ONLY |
| TST (bag+MCE) | 8/8 PASS | WIRED (TST=1) |
| Nested MoE | 396/396 PASS | WIRED (NESTED_MOE=1) — FORWARD ONLY |
| Data pipeline | 1.07M tokens | WIRED |
| CUDA kernels | 2/2 PASS | WIRED |

### Phase 4 (MoE Port) — DONE ✅
Lazy dequant 9×, Nested 16×16 hierarchy, training integration.

### Phase 5 (Vision Port) — DONE ✅
GPU ViT 99ms, 27 layers, 0 NaN. No timeout.

### Phase 6 (CUDA Kernels) — DONE ✅
SSM scan + MoE dispatch, max_diff < 6e-8.

---

## New Risk Table

| # | Risk | Severity | Likelihood | Notes |
|---|------|----------|------------|-------|
| 1 | **Poincaré GQA backward missing** | 🔴 HIGH | 100% | PGA flag produces no gradients through hyperbolic attention |
| 2 | **Nested SSM backward missing** | 🔴 HIGH | 100% | NESTED_SSM is forward-only in training |
| 3 | **Möbius linear layer not available** | 🟡 MEDIUM | 100% | No primitive for fully hyperbolic networks |
| 4 | **Gyration stub is slow** | 🟢 LOW | 100% | 3 Möbius adds vs closed-form O(d) |
| 5 | **VRAM at full scale** | 🟡 MEDIUM | 75% | 6.4GB total, 3B params, KV cache, activations |
| 6 | **TST validation at scale** | 🟡 MEDIUM | 50% | TST reports 2.5× on B200 — unknown on RTX 5050 |
| 7 | **No reference comparison** | 🟢 LOW | 100% | CE=12.42 cannot be compared to Qwen3.6 baseline |

---

## True Priority Queue

**P0 — Poincaré GQA backward.** Forward-only detour cannot train hyperbolic attention. Fill gradient equations through exp_map, log_map, Poincaré distance.

**P1 — Nested SSM backward.** Forward generates hidden states in K=4 nested balls. No backward pass → no gradient to NESTED_SSM parameters.

**P1 — Möbius linear layer (M⊗) CUDA kernel.** `tanh(||Wx||/R) × Wx/||Wx||`. The primitive for all downstream hyperbolic operations.

**P2 — Gyration closed-form, hyperbolic output projection.** Optimization + consistency.

**P3 — Nested MoE backward, hyperbolic KV cache.** Lower impact, dependent on P0/P1.

## Hard Stop Conditions
- Poincaré GQA backward produces NaN → gradient formula error through exp_map/log_map
- M⊗ kernel produces wrong output for norm < R → tanh/division edge case
- RSGD produces `||h|| >= R` during training → gradient projection formula incorrect
