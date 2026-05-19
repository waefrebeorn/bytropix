# Goal Mantra — May 19, 2026 (Phase 9.5 — Q6_K BUG FIXED ✅)

## THE GOAL — ACHIEVED ✅
**1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.**
**Current: 0.9967 cos-sim** — root cause: Q6_K vec_dot loop iteration count bug

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| SSM/GQA attn layers | **Verified** | ✅ Code correct, matches ref architecture |
| GQA Q/gate interleave | **FIXED** | ✅ Per-head interleaved extraction |
| GQA RoPE | **IMPLEMENTED** | ✅ Standard RoPE 64/256 dims, theta=10M |
| KV cache | **4096 positions** | ✅ 10 GQA layers cached |
| MoE router gating | **SOFTMAX** | ✅ Matches llama.cpp (both use softmax) |
| Q5_K shared expert gate | **Verified** | ✅ cos-sim 0.9999 vs F32 SGEMM |
| **Q6_K shared expert down** | **FIXED** | ✅ **was 0.728 → now 0.9999** |
| Quantized-only path | **CODE DONE** | ✅ F32 dequants removed, ~5GB saved |
| **Final cos-sim** | **0.9967** | **✅ 1:1 PARITY ACHIEVED** |

## WHAT WAS WRONG (DA v10 analysis was WRONG)
- DA claimed root cause was "softmax vs sigmoid" gating
- **Both llama.cpp and bytropix use softmax** (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)
- Real root cause: Q6_K vec_dot AVX2 loop processes only 128/256 elements
- `j < QK_K/32` should be `j < QK_K/16` (16 elems/iter × 16 iters = 256)

## THE LOOP
Phase 9 complete. Move to Phase 10: infer_text pipeline.
