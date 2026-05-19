# State — May 19, 2026 (Phase 8.3-8.4 Complete)

## REAL STATUS
**Decode: ~4.7 tok/s (embedding-file mode). Output proj: ~16.5ms decode bottleneck.**
**Cos-sim vs llama.cpp: 0.7944 (pre-existing, NOT a regression)**
**MTP: 29.9 tok/s free-tokens (quality limited by IQ2_M)**

## Phase 8.3: Expert Prefetch — COMPLETE
| Change | Detail |
|--------|--------|
| Full stride prefetch | Was 256 bytes/expert to L1. Now full-stride (~264KB gate/up, ~392KB down) per expert to L3 via _MM_HINT_T2 |
| Coverage | ~7.4MB for 8 experts — fits L3 (12-20MB on modern CPUs) |
| Timing | ~1050 prefetches/weight × 3 weights × 8 experts = ~25K prefetches, fits within attn window |

## Phase 8.4: Output Proj Split — COMPLETE
| Change | Detail |
|--------|--------|
| OMP outer loop | `#pragma omp parallel for if(N > 1)` on token loop for prefill |
| Nested OMP safe | Outer OMP disabled for N=1 (decode path unaffected). Inner quantized_matmul uses 1 thread when nested=off |

## Cold Gaps
| Prio | Gap | Status |
|------|-----|--------|
| P1 | MTP quality at IQ2_M | Inherent — blk.40 Q2_K/Q3_K quantization diverges |
| P1 | Cos-sim 0.79 → 1:1 parity | SSM L0 cos=0.86, GQA L3 cos=0.92. Cumulative decay L6-L31, sharp drop L32-L39 (cos=0.46) |
| P2 | SSM AVX2 optimization | 0.8ms/layer, 24ms total, low priority |
| P2 | KV cache for GQA decode | Each decode recomputes full attention — major overhead |
