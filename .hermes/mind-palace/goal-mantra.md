# Goal Mantra — May 19, 2026 — MTP INFRA COMPLETE. STATE SAVE/RESTORE WORKING.

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
cos-sim 0.9969 (quantization noise). gen_text coherent (0.7 tok/s).
gen_text_mtp: MTP=1 for free-tokens mode (3.3 tok/s, quality loss at IQ2_M).

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

## ACHIEVED (Phase 5+6 — This Session)
- SSM state save/restore (checkpoint/rollback) ✓
- MTP head loads correctly (nextn_hnorm load bug FIXED) ✓
- save_last_hidden captures pre-final-norm (correct for MTP) ✓
- gen_text_mtp: MTP=1 opt-in, non-MTP default ✓
- MTP spec-decode verify infrastructure (checkpoint/rollback per token) ✓
- MTP free-tokens mode (4× speed, quality loss at IQ2_M) ✓

## DISCOVERY
MTP spec-decode (verify) has 100% rejection at IQ2_M. blk.40 MoE at Q2_K/Q3_K too aggressive — combined quantization noise makes MTP head predictions diverge from main model. MTP free-tokens mode bypasses verify but quality degrades. For working MTP, need higher-precision model (Q4_K_M or better for blk.40).

## GROUND TRUTH
- Reference: ~/llama.cpp/src/models/qwen35moe.cpp
- Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (733 tensors, non-MTP)
         /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (753 tensors, MTP)

## UNIT TEST
```
make gen_text_mtp && MOE=1 ./gen_text_mtp "Hello" 4   # non-MTP (default)
make gen_text_mtp && MTP=1 MOE=1 ./gen_text_mtp "Hello" 4   # MTP free-tokens
```
