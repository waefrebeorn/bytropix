# Plan — May 19, 2026 (Phase 11: IQ3_XXS AVX2 ✅ → Phase 12: Output Proj)

## Phase 0-10: DONE ✅
| Phase | Detail | Status |
|-------|--------|--------|
| 0-7 | GQA attn, AVX2 vec_dot, self-contained | ✅ |
| 8 | MoE optimization (IQ2_XXS, OMP, prefetch) | ✅ |
| 9 | Quantized-only path (F32 dequants removed) | ✅ |
| 9.5 | Q6_K vec_dot bug fix (loop iter count) | ✅ FIXED |
| 10 | KV cache 256k (F16, heap attn_weights) | ✅ |

## Phase 11: IQ3_XXS AVX2 Vec Dot — DONE ✅
- 37/40 MoE down weight layers use IQ3_XXS (type 18)
- Ported from llama.cpp AVX2: _mm256_set_epi32 grid lookup, keven_signs_q2xs
- 1.8x speedup: 2.05ms → 1.16ms per MoE layer
- Decode: 7.0 → 9.1 tok/s
- Prefill: 10.5 → 13.4 tok/s
- Cos-sim 0.9965 (FMA accumulation jitter, acceptable)
- 3 layers IQ4_XS still use generic (no AVX2 needed)

## Phase 12: Output Proj Speed [P1 — NEXT]
Output projection: 2048×248320 Q4_K matmul, currently 10ms per token.
- The output proj maps hidden[2048] → logits[248320] via quantized_matmul
- Q4_K type uses generic vec_dot (no AVX2)
- llama.cpp's Q4_K has multiple AVX2 paths: ggml_vec_dot_q4_K_q8_K_avx2
- Porting these would cut output proj from 10ms to ~3-4ms
- Also benefits from blocking/CACHELINE-aware looping

## Phase 13: SSM AVX2 Optimization [P2]
SSM: 1ms/layer × 30 = 30ms total. SSM has:
- attn_qkv: [2048] @ [2048, 8192] (Q4_K)
- attn_gate: [2048] @ [2048, 4096] (Q4_K)  
- Selective scan: 32 heads × 128×128 state (512 dim)
- ssm_out: [4096] @ [4096, 2048] (Q6_K, already AVX2 fixed)
