# WuBuText AI — Devil's Advocate v3: Post-Q4K-Fix Reality Check

## Date: May 13, 2026
## Based on: Fresh runtime verification of all 8 key binaries

---

## Executive Summary

**Previous DA (v2, May 12) claimed** the project was frozen on IQ2 dequantization garbage, NaN from bad dequant, and commented-out CE loss. **All three claims are STALE.** Q4_K dequant bug was the ROOT CAUSE of all NaN and garbage output. After fix:

- train_real CE loss: 6.6e10 → 12.66 (near random baseline 12.42) ✅
- test_moe output: ±1e6 → [-0.028, 0.031] ✅
- All 40 layers forward verified on CPU

**NEW threats have emerged** that the v2 DA didn't see:

| v2 Claim | v3 Reality | 
|----------|-----------|
| Risk 1: attn_qkv split unknown | RESOLVED — SSM forward works from GGUF weights |
| Risk 2: SSM recurrence formula | RESOLVED — verified vs train_real output (12.66 CE) |
| Risk 3: VRAM insufficient | DEFERRED — CPU training at 0.2 tok/s works for testing |
| Risk 5: No BBPE tokenizer | MITIGATED — C tokenizer works, CJK round-trip passes |
| **NEW: GPU weight loading broken** | **🔴 HIGH — bench_e2e produces zeros, not QA'd** |
| **NEW: train_backprop hangs** | **🔴 HIGH — gradient training path non-functional** |
| **NEW: train_gpu wrong loss** | **🟡 MEDIUM — GPU forward gives CE 69 vs 12.66** |

---

## Phase-by-Phase Results

### Phase 0 (GGUF Reader) — DONE ✅
- 13 GGML types supported
- Q4_K/Q5_K dequant fixed (root cause of all NaN)
- All 733 tensors load correctly

### Phase 1 (Embedding Graft) — DONE ✅ (May 12)
- Poincaré ball R=0.956, 95% NN preservation
- Embeddings: 1.9GB file, 248K tokens

### Phase 2 (Attention Port) — DONE ✅ (May 13)
- 40 layers (30 SSM + 10 GQA) forward on CPU
- CE loss 12.66 from wubu_model_forward_from_embd
- GPU kernels exist but PRODUCE ZEROS — NOT USABLE

### Phase 3 (Training Loop) — PARTIAL ⚠️
| Sub-component | Status | 
|--------------|--------|
| Forward pass | ✅ CPU working, CE 12.66 |
| CE loss computation | ✅ streaming 248K vocab |
| Output.weight gradient | ⚠️ train_backprop hangs |
| GPU forward | ⛔ zeros (bench_e2e) or garbage (train_gpu) |
| MoE training | ✅ test_moe passes, 36.6 tok/s |

---

## Risk Table (v3)

| # | Risk | Severity | Likelihood | Phase It Hits | Status Change |
|---|------|----------|------------|---------------|--------------|
| 1 | GPU weight loading broken | 🔴 HIGH | 100% | Phase 3 | NEW — not in v2 |
| 2 | train_backprop hangs | 🔴 HIGH | 100% | Phase 3 | NEW — not in v2 |
| 3 | train_gpu wrong loss | 🟡 MEDIUM | 90% | Phase 3 | NEW — not in v2 |
| 4 | MoE not integrated in training | 🟡 MEDIUM | 50% | Phase 3 | NEW — Q4_K fix made it possible |
| 5 | VRAM insufficient for full training | 🟡 MEDIUM | 75% | Phase 4 | DOWN — CPU training at 0.2 tok/s possible |
| 6 | Tokenizer CJK merge hash collisions | 🟢 LOW | 30% | Phase 3 | DOWN — 58529/524288 (11%), not blocking |
| 7 | SSM/GQA forward correctness | 🟢 LOW | 10% | Phase 2 | RESOLVED — CPU path verified |

---

## True Priority Queue

**P0 — Fix GPU weight loading.** Root cause of bench_e2e zeros and train_gpu wrong loss. `gpu_load_ssm_layer()` in bench.c reads garbage from GGUF. Fix this and both bench_e2e + train_gpu are unblocked.

**P1 — Debug train_backprop hang.** Same compilation as train_real. Likely stdout buffering or OOM. Add fflush + malloc guards.

**P2 — Integrate MoE into training.** test_moe passes. The 256 expert router + shared expert works. Wire into train_real.

**P3 — Verify GPU backward pass.** After P0, GPU forward gives correct CE. Add GPU gradients.

---

## Hard Stop Conditions (NEW)

- bench_e2e still zeros after GPU weight loading fix → GPU kernel or dequant has hidden bug
- train_backprop CE loss > 20 first step → gradient computation or loss function error
- train_gpu CE doesn't converge toward train_real CE → GPU forward is fundamentally different from CPU
