# State — May 19, 2026 (Phase 8 — 8.0 tok/s)

## REAL STATUS
AVX2 IQ2_XXS vec_dot: ✅. OpenMP task dispatch: ✅. Expert prefetch API: ✅.
**Decode: 8.0 tok/s (3.8× from 2.1). Prefill: 10.3 tok/s (1.5× from 6.8).**
MoE: 1.9-2.0ms/layer (↓80% from 10ms). Output proj: 5.7ms. SSM: 0.8ms/layer.

## Phase 8: COMPLETE
| Task | Impact | Detail |
|------|--------|--------|
| AVX2 IQ2_XXS vec_dot | +~20% | Ported from llama.cpp x86/quants.c, 1024-byte keven_signs_q2xs, _mm256_sign_epi8 + _mm256_maddubs_epi16 |
| OpenMP task dispatch | +~200% | `#pragma omp taskgroup` + tasks for 8 experts. Local scratch buffers eliminate atomic. Single parallel region. MoE: 10ms→2ms |
| Expert prefetch API | Plumbing | wubu_moe_forward now returns selected expert indices. Ready for _mm_prefetch integration |

## Bottleneck Distribution (PROFILE=1, decode, 16 threads)
```
MoE (40 layers)         ███████████████████████████████████████  1.9ms × 40 = 76ms (66%)
SSM attn (30 layers)    ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.8ms × 30 = 24ms (21%)
Output projection       ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.7ms (5%)
GQA attn (10 layers)    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~0.5ms × 10 = 5ms (4%)
Norms/overhead          ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~5ms (4%)
```

## Phase 9 Targets
| Prio | Gap | Impact | Status |
|------|-----|--------|--------|
| P1 | Expert prefetch | ~20% MoE reduction | Ready to integrate |
| P1 | SSM attn AVX2 | 24ms total, unoptimized scalar loop | 🔲 |
| P2 | NV64 RDRAM ring buffer | Cache miss latency hiding | Design doc done |
| P2 | cos-sim re-verify | Stale 0.9969 claim | 🔲 |
