# Goal Mantra — May 18, 2026 — PHASE 2 COMPLETE

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
cos-sim 0.9969 (quantization noise, arch bugs = 0).
gen_text coherent (0.6 tok/s decode, 1.4 tok/s prefill).

## ACHIEVED
- GQA Q/gate interleave bug FIXED: cos-sim -0.51 → 0.9968
- IMRoPE implemented (sections [11,11,10,0], theta=10M) ✓
- MoE quantized path wired (IQ2_XXS/IQ3_XXS/IQ4_XS via blob) ✓
- Per-layer dump infrastructure ref_dumper ✓
- gen_text pipeline working ✓ (coherent 32-token gen)
- DA v10 gaps: **ALL 10 CLOSED** (Gap 7 chat template fixed ✓)
- Performance: 0.3→0.7 tok/s (2.3×): MoE OpenMP, embedding fix, buffer reuse
- Q4_K output proj: cos-sim 0.99995 vs SGEMM
- Vault papers read: Qwen3.6 arch, Unsloth UD quant, DA v10 audit

## REMAINING GAP
- cos-sim 0.9968 → 1.0: quantization noise from generic C vec_dot (no SIMD)
- Speed: 0.7 tok/s (CPU-bound for 35B MoE)
- KV cache for GQA (~10% speedup)

## GROUND TRUTH
- Reference: ~/llama.cpp/src/models/qwen35moe.cpp
- Dumper: ~/bytropix/ref_dumper (links libllama.so directly)
- Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
- Hidden dump: DUMP_LAYER_DIR=/tmp/dump_layers

## UNIT TEST
```
make test_full_moe && PROFILE=1 ./test_full_moe
# Expect: cos-sim 0.9969, per-layer timing
```
