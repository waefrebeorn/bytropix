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
- **SSSE3/SSE4.1 vec_dot**: Q4_K, Q5_K, Q6_K SSE intrinsics. cos-sim 0.9970 ↑
- **GQA KV cache**: decode attends to all tokens, not just self (correctness fix) ✓

## REMAINING GAP
- cos-sim 0.9970 → 1.0: IQ2_XXS lookup-table vec_dot (need SIMD gather)
- Speed: 0.7 tok/s on CPU — memory bandwidth bound (10.7GB/step, DDR5 ~50GB/s)
- MTP model not yet wired (blk.40 with nextn projections)

## NEXT TASK
Island Boy + MTP speculative decode architecture.
1. Load MTP model (blk.40 GQA+MoE+nextn)
2. Batch-process tokens per layer (B=4, amortize weight load)
3. 5-token startup lag, then synchro speedup
4. Target: 2-3 tok/s via batch decode + spec-decode

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
