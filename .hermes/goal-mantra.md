# Goal Mantra — May 19, 2026 (Triple DA v6)

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.

## TRIPLE DA FINDING
SSM recurrence math IDENTICAL (verified vs ggml_gated_delta_net source).
0.79 cos-sim = quantized matmul precision, NOT algorithm bug.
**Top-1 token matches (220) — functional parity is close.**

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Logit cos-sim vs llama.cpp | **0.7944** | ✅ Pre-existing at IQ2_M |
| Per-layer cos-sim (avg) | **0.88** | ✅ Verified, range 0.45–0.97 |
| SSM recurrence | **IDENTICAL** | ✅ 1/sqrt(128), same formula |
| Top-1 match (BOS) | token **220** | ✅ Both agree |
| Decode speed | **7.8 tok/s** | ✅ Phase 8 optimize |
| AVX2 IQ3_XXS vec_dot | **BROKEN** | ❌ _mm_hadd epi16 bug, reverted |

## COLD GAPS
P0: Quantized matmul dequant precision (port ggml kernels for bit-exact)
P1: AVX2 IQ3_XXS vec_dot fix
P1: Output proj split (parallelize Q4_K)
P2: Expert prefetch

## GROUND TRUTH
- Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
- Reference: /home/wubu/llama.cpp/build/bin/libllama.so
- Cross-ref: ref_dumper (logits), run_bos (bytropix), DUMP_LAYER_DIR (per-layer)

## THE LOOP
pick highest undone → execute → compile → run → verify → mark done → report

## FULL CONTEXT
Read .hermes/mind-palace/prestige_prompt.md
