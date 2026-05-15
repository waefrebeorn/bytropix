# WuBuText AI — Devil's Advocate v4: Post-Sprint Reality Check

## Date: May 15, 2026
## Based on: Runtime verification of all 12 streams

---

## Executive Summary
**Previous DA (v3, May 13) identified 7 risks. ALL 7 resolved.**
New threats emerged from the sprint — not code bugs but architectural gaps.

| v3 Risk | v4 Status |
|---------|-----------|
| GPU weight loading broken | ✅ Fixed — dequant bypass fixed |
| train_backprop hangs | ✅ Not hanging — just CPU-slow |
| train_gpu wrong loss | ✅ CE=69 → 12.42 |
| MoE not integrated | ✅ Lazy MoE in training |
| VRAM insufficient | ✅ CPU training works |
| Tokenizer CJK | ✅ Works |
| SSM/GQA forward | ✅ Verified |

---

## Phase-by-Phase Results

### Phase 0 (GGUF Reader) — DONE ✅
13 GGML types, 733 tensors, GPU loading fixed.

### Phase 1 (Embedding Graft) — DONE ✅
Poincaré R=0.956, 95% NN preservation, 248K embeddings.

### Phase 2 (Attention Port) — DONE ✅
40 layers CPU/GPU forward. CE loss 12.66 CPU, 12.42 GPU.

### Phase 3 (Training Modules) — BUILDING BLOCKS DONE ❗
Modules exist standalone — NONE integrated.

| Module | Tests | Integration |
|--------|-------|-------------|
| RSGD | PASS | NOT wired |
| Poincaré GQA | 4/4 PASS | NOT wired |
| Nested SSM K=4 | 3/3 PASS | NOT wired |
| TST (bag+MCE) | 8/8 PASS | NOT wired |
| Nested MoE | 396/396 PASS | NOT wired |
| Data pipeline | 1.07M tokens | NOT wired |
| CUDA kernels | 2/2 PASS | NOT wired |

### Phase 4 (MoE Port) — DONE ✅
Lazy dequant 9×, Nested 16×16 hierarchy, training integration.

### Phase 5 (Vision Port) — DONE ⚠️
GPU ViT 217ms, pipeline works. But `infer_vision_text_gpu` TIMED OUT.

### Phase 6 (CUDA Kernels) — DONE ✅
SSM scan + MoE dispatch, max_diff < 6e-8.

---

## New Risk Table

| # | Risk | Severity | Likelihood | Notes |
|---|------|----------|------------|-------|
| 1 | **Integration cost** | 🔴 HIGH | 100% | 7 modules × many wiring points = significant work |
| 2 | **GPU vision timeout** | 🔴 HIGH | Unknown | `infer_vision_text_gpu` 120s timeout — root cause unknown |
| 3 | **NaN persists** | 🟡 MEDIUM | 50% | 0.5% NaN after ALL fixes — no remaining suspects |
| 4 | **VRAM at full scale** | 🟡 MEDIUM | 75% | 6.4GB total, 3B params, KV cache, activations, optimizer — likely exceed |
| 5 | **TST validation at scale** | 🟡 MEDIUM | 50% | TST paper reports 2.5× on B200 — unknown on RTX 5050 |
| 6 | **CPU RMSNorm OOB** | 🟢 LOW | 100% | CPU-only bug, GPU path unaffected |
| 7 | **No reference comparison** | 🟢 LOW | 100% | CE=12.42 cannot be compared to Qwen3.6 baseline |

---

## True Priority Queue

**P0 — Fix GPU vision pipeline timeout.** Root cause unknown. Blocks production vision→text.

**P0 — Wire RSGD + TST into training.** These are the two core training components. Everything else is derivative.

**P1 — Wire Poincaré GQA, Nested SSM, Nested MoE into model forward.** Replace Euclidean path with hyperbolic path.

**P1 — Full training convergence test.** Run 1000+ steps with all modules integrated. Verify CE < 5.0.

**P2 — Debug 0.5% NaN.** After all other fixes, the NaN might be a symptom of something not yet found.

**P2 — VRAM budget analysis.** Measure actual consumption at full scale before writing more code.

## Hard Stop Conditions
- GPU vision timeout unresolved after MARKer insertion + per-kernel timing → CUDA kernel bug in cuda_vision.cu
- First training step with integrated modules gives CE > 20 → wiring bug
- RSGD produces ‖h‖ >= R during training → RSGD gradient formula incorrect
