# Plan — May 19, 2026 (Phase 8 Active)

## Phase 0-7: DONE
- GQA attn stack buf + AVX2 FMA
- AVX2 vec_dot Q4_K/Q5_K/Q6_K (256-bit)
- Llama deps killed (self-contained vec_dot)
- NV64 RDRAM design doc written

## Phase 8: MoE Optimization [ACTIVE]
| Sub-phase | Status | Detail |
|-----------|--------|--------|
| 8.1 AVX2 IQ2_XXS vec_dot | ✅ DONE | Ported from llama.cpp x86/quants.c. 1024-byte keven_signs_q2xs table, _mm256_sign_epi8, _mm256_maddubs_epi16 |
| 8.2 OpenMP task dispatch | ✅ DONE | #pragma omp taskgroup + task for 8 experts. Local scratch buffers eliminate atomic. Single parallel region. |
| 8.3 Expert prefetch | 🔜 NEXT | Predict next layer's expert indices, prefetch weight data to L1/L2 |
| 8.4 Output proj split | 🔜 P0 | Split Q4_K output proj across 16 threads (248K columns) |

## Phase 9: Expert Prefetch [NEXT — P0]
- Predict next layer's expert routing from current hidden state
- Preload weights for predicted experts to L2 via _mm_prefetch
- 2-level prefetch window (current + next layer)
- Expected: ~20% reduction in MoE decode time

## Phase 10: Output Proj Split [P0]
- Current: single-threaded Q4_K matmul [2048] × [2048, 248320]
- Fix: parallelize across columns with OMP
- Expected: 5.7ms → ~1ms

## Phase 11: NV64 RDRAM Ring Buffer [P1]
- ring_slot_t[64] with atomic head/tail
- Prefetch agent thread (graduated T2→T1→T0)
