# DA v8 — Devil's Advocate Full Audit (May 17 PM)

## Phase 1: Claims Audit — Current state.md

### Claim 1: "MoE expert dequant (IQ2_XXS gate/up, IQ3_XXS down, IQ4_XS down) ✅"
**Source:** code + MoE output rms=0.25 after fix
**Trust:** HIGH — verified empirically (rms dropped from 690k to 0.25)
**Verify:** moe_out_00.bin rms=0.25, max=2.2 ✓
**BUT:** IQ4_XS dequant function is NEW (written from reading llama.cpp source, never verified with test data or against reference). Only used at L34/38/39 — not hit in single-layer tests.

### Claim 2: "SSM QKV/conv/recurrence verified vs Python (cos_sim=1.0) ✅"
**Source:** tools/verify_ssm.py
**Trust:** **MEDIUM — survivorship bias detected**
**Verify:** The script compares Python SSM output against C code's DUMP_SSM_VAL dump. Both compute the SAME formula from the SAME weights. This verifies Python and C implementations ARE CONSISTENT — NOT that either matches llama.cpp reference.
**Key gap:** No comparison against llama.cpp's SSM output. The formula itself may differ from reference.
**Recommendation:** Add llama.cpp SSM output dump and compare.

### Claim 3: "Shared expert output reasonable (rms=0.51) ✅"
**Source:** shared_out.bin dump from L0
**Trust:** HIGH — empirically valid
**Verify:** shared_out contains reasonable values ✓

### Claim 4: "All 40 layers process without crash ✅"
**Source:** Full run output
**Trust:** HIGH — confirmed by 135s prefill completing
**Verify:** Full run shows all 40 layers processed ✓

### Claim 5: "SSM L0 cos_sim=0.40 vs reference"
**Source:** Layer 0 residual comparison
**Trust:** MEDIUM — comparison point may be wrong
**Problem:** The comparison uses our L0 post-SSM residual vs reference's L0 post_moe (which includes MoE). Our dump was POST-MoE for L0 (after post-MoE dump was added in the code). Need to verify exact comparison point.
**Verify:** Was cos_sim=0.40 measured against pre-MoE or post-MoE dump? With IQ3_XXS fix, MoE output is now correct, so post-MoE should match better if MoE was the only issue.

### Claim 6: "IQ3_XXS block size: 98 bytes (NOT 104)"
**Source:** llama.cpp ggml-common.h struct definition
**Trust:** HIGH — confirmed by llama.cpp source
**Verify:** `sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8) = 2 + 96 = 98` ✓
**BUT:** Our dequantize_iq3_xxs_row uses IQ3_XXS_BLOCK_SIZE=98 but processes with INTERNAL structure that may differ from llama.cpp's. The comment says "qs[64] + scales_and_signs[32]" but llama.cpp uses qs[96]. The internal interpretation may produce different output even though block size is correct.

## Phase 2: Architecture Cross-Check vs Papers

### SSM Formula vs Reference
Our SSM recurrence (Gated DeltaNet):
```
h_t = h_{t-1} * exp(alpha * softplus(dt + dt_bias) * ssm_a)  // state decay
      + k_t * (v_t - h_{t-1} @ k_t) * beta_t                    // state update
out_t = h_t @ q_t                                               // readout
```
Need to verify this matches the reference's Gated DeltaNet implementation.

### D_MODEL=2048 — Hardcoded Assumption
Not read from GGUF metadata. Works empirically (tensor sizes match), but should be verified.

### MoE Router — Uses softmax, not sigmoid
Our code uses softmax routing (lines 388-393 of infer_text.c). DeepSeekMoE/Qwen3 use normalized sigmoid gating. This may cause routing collapse during training, but for inference the top-K selection is similar.

## Phase 3: Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **SSM formula differs from reference** | Inference always wrong | Medium | Dump SSM state/output from llama.cpp and compare layer-by-layer |
| **IQ4_XS dequant untested** | L34/38/39 produce wrong MoE output | Medium | Test with one of those layers specifically |
| **SSM output weight dequant wrong** | L0 SSM output projection wrong | Medium | Verify ssm_out.weight tensor type and dequant |
| **Model config assumptions stale** | Wrong D_MODEL/D_FF/etc for model | Low | Verify GGUF metadata vs hardcoded constants |
| **IQ3_XXS internal format mismatch** | Down weights numerically wrong despite correct block size | Low-Medium | Compare dequantized weight values against llama.cpp |
| **DA v3's "IMRoPE is root cause" was wrong** | Misdiagnosis wasted prior sessions | HIGH — CONFIRMED | SSM doesn't use RoPE, so SSM divergence is a separate bug |

## Phase 4: Stale Claim Sweep

- "IMRoPE is root cause" (DA v3/v7) → **STALE.** SSM doesn't use RoPE. SSM-only cos_sim=0.018 demonstrates independent bug.
- "SSM formula verified correct" (state.md v15) → **MISLEADING.** Only verified Python=C consistency, not correctness vs reference.
- "IQ4_XS dequant added" → **UNTESTED.** Written from reference, never run against real tensor data.

## Phase 5: Fresh Priorities

**P0 — Verify SSM formula against llama.cpp:**
1. Dump SSM recurrence inputs (Q, K, V, gate, beta) and outputs (h, out) from our code and ref
2. Compare to find exact divergence point
3. Test with and without ssm_out projection to isolate output proj vs recurrence

**P1 — Verify model config from GGUF metadata:**
Read hidden_size, n_heads, n_layers, n_experts from GGUF KV pairs, compare against hardcoded constants.

**P2 — Test IQ4_XS dequant on actual IQ4_XS tensor:**
Test single-layer inference on L34 specifically.
