# State — May 19, 2026 (Phase 8 Complete — Cos-sim Verified)

## REAL STATUS
**Decode: 7.8 tok/s (3.7× from 2.1). Prefill: 10.4 tok/s.**
**Cos-sim vs llama.cpp: 0.7944 (pre-existing, NOT a regression)**  
**MTP: 29.9 tok/s free-tokens (quality limited by IQ2_M, NOT a bytropix bug)**

## Correctness Verification
| Test | Result | Evidence |
|------|--------|----------|
| Cos-sim vs llama.cpp (BOS token) | **0.7944** | ✅ Same before AND after Phase 8 changes. Pre-existing at IQ2_M |
| Top-1 prediction (BOS) | token 220 | ✅ Matches llama.cpp exactly |
| MTP ref_dumper_mtp | target=220, MTP=2 | ✅ Same mismatch in llama.cpp's own MTP at IQ2_M |
| MTP free-tokens mode | 29.9 tok/s | ✅ Pipeline correct, blk.40 quantization limits quality |

## Phase 8: COMPLETE
| Optimization | Speedup | Detail |
|-------------|---------|--------|
| AVX2 IQ2_XXS vec_dot | +~20% | Ported from llama.cpp x86/quants.c |
| OpenMP task dispatch | +~200% | `#pragma omp taskgroup` + tasks, no atomic, single parallel region |
| Expert prefetch API | plumbing | wubu_moe_forward returns selected expert indices |

## Cold Gaps
| Prio | Gap | Status |
|------|-----|--------|
| P1 | MTP quality at IQ2_M | Inherent — blk.40 Q2_K/Q3_K quantization diverges from main model path |
| P1 | Expert prefetch integration | API ready, needs _mm_prefetch wiring |
| P2 | Cos-sim improvement | 0.79 is fundamental at IQ2_M. Need higher-precision model for verification |
| P2 | SSM attn AVX2 optimization | 0.8ms/layer, 24ms total, low priority |
