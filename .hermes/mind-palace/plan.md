# Plan — May 19, 2026 — PHASES 0-6 DONE. MTP INFRA + STATE SAVE/RESTORE COMPLETE.

## Phase 0: CORE INFERENCE ✓
- GQA Q/gate interleave fixed (cos-sim -0.51 → 0.9968)
- All 40 layers verified: cos-sim > 0.995
- MoE quantized path wired (IQ2_XXS/IQ3_XXS/IQ4_XS via blob)
- Shared expert quantized path (Q5_K/Q6_K)
- Per-layer dump via ref_dumper

## Phase 1: DA v10 Closure ✓
- All 10 gaps closed (cos-sim 0.9969, chat template fixed)

## Phase 2: Performance Optimization ✓
- MoE OpenMP: 3× speedup (44ms→15ms per layer)
- Embedding file: opened once, closed at end
- Decode: 0.3→0.7 tok/s (2.3×)

## Phase 3: SIMD vec_dot ✓
- Q4_K/Q5_K: SSSE3 _mm_maddubs_epi16
- Q6_K: SSE4.1 _mm_cvtepi8_epi16
- Cos-sim: 0.9968 → 0.9970

## Phase 4: KV Cache ✓
- K/V cache per GQA layer [10][4096][512]
- Append-only: decode attends to ALL cached positions
- cos-sim T=1 identity preserved

## Phase 5: State Save/Restore ✓
- SSM state + conv state checkpoint/rollback
- GQA cache len save/restore
- MTP head cache len save/restore
- Lazy allocation (only when checkpoint first called)
- Used for MTP spec-decode rollback

## Phase 6: MTP Speculative Decode ✓
### Infrastructure ✓
- SSM state save/restore for verify rollback
- Batch-forward draft tokens through 40 layers (sequential verify)
- Per-token checkpoint → rollback on reject
- MTP free-tokens mode (no verify, emit MTP outputs directly)
- MTP=1 opt-in environment variable
### Known Limitation
- At IQ2_M: 100% verify rejection, MTP free-tokens quality degraded
- Need higher-precision model (Q4_K_M+ for blk.40) for working MTP
### MTP Head Loading ✓
- nextn_hnorm load bug fixed (was found but never allocated)
- blk.40 GQA+MoE (Q5_K/Q2_K/Q3_K/Q6_K)
- eh_proj dequantized to F32 at init
- KV cache for blk.40 GQA

## Phase 7: Hardware Saturation [NEXT]
- AVX2/AVX-512 vec_dot paths
- NUMA-aware thread scheduling
- Quantized scatter/gather for IQ MoE
- _mm_prefetch weight prefetching
- OpenMP on GQA attention (softmax + score loop)
