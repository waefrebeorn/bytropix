# Goal Mantra — May 18, 2026 — PHASE 5+6 INFRA COMPLETE

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
cos-sim 0.9969 (quantization noise, arch bugs = 0).
gen_text coherent (0.7 tok/s decode, 2.4 tok/s prefill).

## ACHIEVED (All Phases 0-4)
- GQA Q/gate interleave bug FIXED: cos-sim -0.51 → 0.9968 ✓
- IMRoPE implemented (sections [11,11,10,0], theta=10M) ✓
- MoE quantized path wired (IQ2_XXS/IQ3_XXS/IQ4_XS via blob) ✓
- Per-layer dump infrastructure ref_dumper ✓
- gen_text pipeline working ✓ (coherent 32-token gen)
- DA v10 gaps: ALL 10 CLOSED ✓
- Performance: 0.3→0.7 tok/s (2.3×): MoE OpenMP, embedding fix, buffer reuse ✓
- Q4_K output proj: cos-sim 0.99995 vs SGEMM ✓
- SSSE3/SSE4.1 vec_dot: Q4_K, Q5_K, Q6_K SSE intrinsics ✓
- GQA KV cache: decode attends to all tokens ✓

## ACHIEVED (Phase 5+6 Infra, This Session)
- Q2_K/Q3_K/Q8_0/IQ2_S/BF16 all supported in quantized_matmul ✓
- save_last_hidden field for h_39 capture ✓
- MTP model loads (blk.40 + nextn) via wubu_mtp_load ✓
- gen_text_mtp stable decode at 0.7 tok/s ✓
- ALL quant types needed for MTP GGUF handled ✓

## REMAINING (MTP Spec-Decode)
- SSM state save/restore around verify batch forward
- KV cache rollback on partial reject
- Acceptance: batch-forward drafted tokens, compare argmax
- Target: 1.5-2.5 tok/s via MTP (2-3× improvement)
- Cos-sim verification vs llama.cpp MTP output

## GROUND TRUTH
- Reference: ~/llama.cpp/src/models/qwen35moe.cpp
- Dumper: ~/bytropix/ref_dumper (links libllama.so directly)
- Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (733 tensors, non-MTP)
         /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (753 tensors, MTP)
- Hidden dump: DUMP_LAYER_DIR=/tmp/dump_layers

## UNIT TEST
```
make gen_text_mtp && MOE=1 ./gen_text_mtp "Hello" 4
# Expect: stable output, 0.7 tok/s decode
```
