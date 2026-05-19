# State — May 19, 2026 (Phase 11 — IQ3_XXS AVX2 ✅)

## REAL STATUS
- **Cos-sim vs llama.cpp: 0.9965** — 1:1 parity ✅ (AVX2 FMA diff from generic, acceptable)
- **KV cache: 262144 positions, F16 format** ✅ (memory: 10GB→5GB)
- **gen_text pipeline working** — 9.1 tok/s decode, 13.4 tok/s prefill
- **Q6_K vec_dot bug FIXED** ✅
- **IQ3_XXS AVX2 vec_dot ported from llama.cpp** ✅

## What's Hot
| Component | Time/token | Priority |
|-----------|-----------|----------|
| MoE (40 layers × 1.2ms) | ~48ms (47%) | DONE — IQ3_XXS AVX2 |
| SSM/GQA (40 layers × 1ms) | 40ms (39%) | P2 |
| Output proj (1 × 10ms) | 10ms (10%) | P1 |

## KV Cache Changes
- GQA_MAX_CTX: 4096 → 262144
- Format: F32 → F16 (via KV_CACHE_F16=1)
- Allocation: using kv_cache_alloc_size() for type-agnostic sizing
- Access: kv_cache_read_head/write_head helpers convert F16↔F32

## IQ3_XXS AVX2
- Ported from llama.cpp ggml-cpu/arch/x86/quants.c (ggml_vec_dot_iq3_xxs_q8_K)
- Uses keven_signs_q2xs table (NOT ksigns_iq2xs — confirmed correct in llama.cpp generic)
- 256-bit grid lookup via _mm256_set_epi32(iq3xxs_grid[q3[i]], ...)
- 64 elements per inner loop iteration (2× Q32 blocks)
- FMA accumulation: _mm256_fmadd_ps for each block
- Speed: 1.16ms/layer vs 2.05ms generic (1.8x)
- Cos-sim 0.9965 vs 0.9967 generic (negligible FMA accumulation difference)

## Cold Gaps
| Prio | Gap | Status |
|------|-----|--------|
| ✅ P0 | **IQ3_XXS AVX2 vec_dot** | ✅ DONE — 1.8x speedup |
| P1 | Output proj speed (10ms) | Q4_K 2048×248320 — blocked matmul? |
| P2 | SSM AVX2 optimization | Future |
