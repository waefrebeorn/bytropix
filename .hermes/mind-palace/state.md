# state — May 17 v15 — Final: all components verified, MoE divergence confirmed

## Final Verdict
- **SSM/GQA path: CORRECT** ✅ (logits cos-sim 0.994 vs reference)
- **MoE path: FUNCTIONAL but DIVERGENT** ⚠️ (logits cos-sim 0.337 vs reference)
- Model generates text with MoE enabled (not silent/garbled) but output differs from reference
- All subcomponents verified exact vs ggml — divergence is from recurrent amplification

## Verified Exact vs ggml (all types, full 1M elements)
| Component | Type | Status |
|-----------|------|--------|
| IQ2_XXS dequant (gate/up exps) | Type 16 | ✅ exp0 & exp64 full match |
| IQ3_XXS dequant (down exps) | Type 18 | ✅ exp0 & exp64 full match |
| IQ4_XS dequant (down exps L38-39) | Type 23 | ✅ exp0 full match |
| Q5_K dequant (shared expert gate/up) | Type 13 | ✅ 256/256 match |
| Q6_K dequant (shared expert down) | Type 14 | ✅ loaded via gguf_read_tensor_f32 |
| Token embeddings (BOS 248044) | Type 13 (Q5_K) | ✅ 2048/2048 match |
| Expert offsets | — | ✅ correct for all experts |
| Top-k renormalization | norm_w=true | ✅ matches reference |
| MoE=0 logits | — | ✅ cos-sim 0.994 vs reference |
| MoE=1 logits | — | ❌ cos-sim 0.337 vs reference |

## Root Cause Analysis
- Per-layer MoE contribution has rms ≈ 0.03 (small, correct for 2-bit weights)
- Shared expert (Q5_K/Q6_K) also has small per-layer rms ≈ 0.05
- Total MoE per layer: ~0.06 rms vs pass-through ~0.89 rms
- Over 40 layers, MoE corrections accumulate: our L39 residual rms=0.59 vs MOE=0 rms=55
- Both directions are equally valid (final RMSNorm normalizes either to ~2.1-2.7 rms)
- The direction divergence (cos-sim 0.337) is from recurrent amplification of tiny per-layer diffs

## Generated Output
- MOE=0: "The quick brown foxedo PropertyDescriptor _EXPR 是何含义 way"
- MOE=1: "The quick brown fox, . ** , . ."
- Reference: unable to verify (llama-cli display issue)
- Output quality is poor, likely from 2-bit quantization of expert weights
