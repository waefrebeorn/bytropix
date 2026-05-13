# Devil's Advocate: WuBuText AI v2 — Post-Proofread Full Scan

**Date:** May 12, 2026 (v2)
**Based on:** Master Implementation Plan v2, all 5 phases proofread, math optimization roadmap written

## Executive Summary

**Claim:** We can build Qwen3.6-35B-A3B → Pure C + CUDA with hyperbolic grafting in ~30-40 steps,
running on RTX 5050 6.4GB VRAM, by extracting pretrained weights and translating geometry.

**Phase 1 result:** ✅ 95% NN preservation at R=0.956 — thesis confirmed.

**Status check:** All docs proofread, math roadmap written, fresh plan in place.
9 corrections found in the original docs — all fixed.

---

## Risk 1: The attn_qkv Split is the Real Blocker — Not the SSM Math

**Severity:** 🔴 HIGH | **Likelihood:** 90%

We don't know how `attn_qkv.weight [2048, 8192]` decomposes. The shape doesn't
cleanly sum to any obvious Q+K+V split for 16 Q heads + 2 KV heads + 16 linear Q heads
+ 32 linear V heads. The total we'd expect is 11264, but only 8192 dims exist.

Three possibilities:
1. **Some projections are computed from x directly** (separate weight matrices not fused into attn_qkv) — most likely
2. **Head dim is smaller for some heads** — possible but unlikely
3. **The split includes extra projection heads** — unknown

**Until we read `llama.cpp/src/models/qwen35.cpp`, Phase 2 cannot produce a correct forward pass.**

**Fix:** Must trace through llama.cpp source. This is the #1 action item right now.

---

## Risk 2: We Don't Know the SSM Recurrence Formula

**Severity:** 🟡 MEDIUM | **Likelihood:** 70%

We know it's Mamba2-style (not DeltaNet) from tensor names, but the EXACT formula
matters for the hyperbolic gyration replacement. Mamba2 uses:

```
h[t] = exp(A·dt) ⊙ h[t-1] + (1 - exp(A·dt)) ⊙ B·x[t]
```

But the Qwen3.5 variant may differ. There may be:
- Different gating: alpha/beta vs A/B
- Different activation: silu vs sigmoid for gate
- Different dt computation: softplus vs sigmoid
- conv1d kernel size=4 with depthwise vs regular

**Each difference changes the hyperbolic gyration formula.**

**Fix:** Read llama.cpp `ggml-ssm.h` and `qwen35.cpp` to get the EXACT formula.
Then implement with matching pseudocode. The hyperbolic variant is derived from this.

---

## Risk 3: 6.4GB VRAM is NOT Enough — Even with 16 Layers

**Severity:** 🔴 HIGH | **Likelihood:** 75%

Updated estimate using ACTUAL tensor sizes from GGUF:

| Category | 16 Layers | 40 Layers |
|----------|-----------|-----------|
| Attention QKV (Q8_K) | 16 × 2048×8192×1 = 268MB | 671MB |
| SSM params (F32) | 16 × ~3MB = 48MB | 120MB |
| MoE experts (IQ2_XS/IQ1_S) | 16 × 256×~1.5MB = 6GB | 15GB |
| Shared expert (Q8_K) | 120MB | 120MB |
| Norms/router (F32) | ~4MB | ~10MB |
| **Weight total** | **~6.4GB** | **~15.9GB** |
| KV cache (4096) | 268MB | 670MB |
| Activations (B=2) | ~1GB | ~2.7GB |
| **Peak VRAM** | **~7.7GB** | **~19GB** |

**16 layers still exceeds 6.4GB.** Even with CPU layer swapping, the MoE experts
(6GB for 16 layers) dominate. IQ2_XS is "2-bit" but the dequant overhead means
we need working memory.

**Hard choices:**
1. **Only keep 8 layers on GPU** — swap during training, significant slowdown
2. **Use Qwen3.5-9B (dense, no MoE)** — 9B dense is still ~1.5GB in Q4_K, but no MoE overhead
3. **Train on CPU** — ~2 tok/s, impractical for anything beyond loss testing
4. **Buy more VRAM** — RTX 5050 isn't upgradeable

**Recommendation:** Phase 2 (forward pass only) works on CPU. Phase 3 (training)
needs the VRAM problem solved. Accept that training will be hybrid (CPU offload
for weights, GPU for compute).

---

## Risk 4: The MoE Quantization is Too Aggressive for Hyperbolic Routing

**Severity:** 🟡 MEDIUM | **Likelihood:** 45%

The expert weights are IQ2_XS (2-bit gate/up) and IQ1_S (1-bit down). At these
levels, each expert's effective capacity is ~180KB for the gate projection.
Comparing: a single attention tensor (Q8_K, 2048×8192) is 16MB.

**The experts are tiny.** Their output is dominated by quantization noise.
If we route based on hyperbolic distance to centroids computed from these
heavily quantized weights, the routing signal may be lost in noise.

**Worst case:** The hyperbolic router routes to experts that are "closest in
embedding space" but dequantization noise makes their actual output garbage.

**Fix:** Keep the Euclidean router as a fallback for Phase 4. The hyperbolic
router should be an ADDITIONAL scoring term, not a replacement.

---

## Risk 5: No BBPE Tokenizer (Unchanged — Still #1 Blocker for Training)

**Severity:** 🔴 HIGH | **Likelihood:** 90%

Without BBPE tokenization, the Phase 1 embeddings are unusable for real data.
Random token IDs for testing? Works for verifying math. Actual training? Dead stop.

**The tokenizer work can start NOW (parallel to Phase 0/2).** It doesn't depend
on anything else. Recommended approach:

1. **Python workaround** (day 1): Call Hugging Face tokenizers via subprocess
2. **C implementation** (week 1): GPT-2 BPE using merges from GGUF

---

## Risk 6: The ssm_dt.bias is [32] — What Does It Mean for Time Steps?

**Severity:** 🟢 LOW | **Likelihood:** 30%

`ssm_dt.bias` shape [32] matches `ssm_time_step_rank = 32` in config.
This suggests dt is NOT element-wise per hidden dimension (4096), but
projected through a rank-32 bottleneck first.

This is a common pattern in Mamba2: the time step delta is first projected
to a smaller rank, then expanded back. The bias [32] is at the projection
level, not the final dt.

**Impact on hyperbolic gyration:** The gyration `A_bar = exp(-exp(A) * dt)` uses
this projected dt. Since DT is scalar per group (32 groups of 128 hidden dims),
the hyperbolic equivalent would also use group-wise Möbius addition.

**This is fine — the hyperbolic extension is the same regardless of dt dimensionality.**

---

## Risk 7: The Nested MoE Tree May Not Improve Over Flat Routing

**Severity:** 🟡 MEDIUM | **Likelihood:** 50%

The 2-level tree (16 groups × 16 experts) saves computation but may lose
routing quality. In flat routing, each token can route to any 8 of 256 experts.
In a tree, a wrong choice at level 1 means the token can never reach the
correct experts in the chosen group's subtree.

**Evidence from MoE literature:** Hierarchical routing generally performs
worse than flat for the same number of experts, but is faster.

**Fix:** Keep flat routing as the default option. Only use hierarchical routing
for inference (speed benefit) or as a regularization technique (training penalty
if hierarchical routing diverges from flat).

---

## Updated Risk Table

| # | Risk | Severity | Likelihood | Phase It Hits | Changes Since v1 |
|---|------|----------|------------|---------------|------------------|
| 1 | attn_qkv split unknown | 🔴 HIGH | 90% | Phase 2 | NEW — was implicit in "SSM implementation unknown" |
| 2 | SSM recurrence formula | 🟡 MEDIUM | 70% | Phase 2 | Split from Risk 1 — the math, not the weights |
| 3 | VRAM insufficient | 🔴 HIGH | 75% | Phase 3 | Updated with actual tensor sizes — worse than v1 estimate |
| 4 | MoE quantization noise | 🟡 MEDIUM | 45% | Phase 4 | NEW — specific to hyperbolic routing performance |
| 5 | No BBPE tokenizer | 🔴 HIGH | 90% | Phase 3 | Unchanged — v1's #1 blocker |
| 6 | ssdm_dt bias shape | 🟢 LOW | 30% | Phase 2 | NEW — detail from config reading |
| 7 | Nested tree vs flat routing | 🟡 MEDIUM | 50% | Phase 4 | NEW — explicit about tree vs flat risk |

## Go/No-Go Verdict

**GO ✅ — with conditions:**

1. **Phase 0 first:** Read llama.cpp `qwen35.cpp` before writing any Phase 2 code
2. **Tokenizer starts parallel:** Don't wait for Phase 2 — start now
3. **VRAM acceptance:** Accept that full training is impossible on 6.4GB.
   Design for: forward pass verification on CPU/GPU, training with CPU offload.
4. **SSM first, hyperbolic second:** Each SSM layer type (SSM, GQA) gets a
   Euclidean reference BEFORE the hyperbolic variant. This proves the math works.

**Hard stop conditions:**
- Phase 2 forward pass differs from llama.cpp by >0.05 RMS → STOP, read source again
- Phase 3 loss > 10.0 after 1000 steps with Euclidean reference → STOP, data/optimizer bug
- Phase 4 router entropy < 1.0 after 500 steps → STOP, routing collapse
