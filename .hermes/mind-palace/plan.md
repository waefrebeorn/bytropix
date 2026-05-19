# Plan — May 19, 2026 (Phase 8.3-8.4 Complete)

## Phase 0-7: DONE
- GQA attn stack buf + AVX2 FMA
- AVX2 vec_dot Q4_K/Q5_K/Q6_K (256-bit)
- Llama deps killed (self-contained vec_dot)
- NV64 RDRAM design doc written

## Phase 8: MoE Optimization [8.3-8.4 DONE]

| Sub-phase | Status | Detail |
|-----------|--------|--------|
| 8.1 AVX2 IQ2_XXS vec_dot | ✅ DONE | Ported from llama.cpp x86/quants.c |
| 8.2 OpenMP task dispatch | ✅ DONE | omp taskgroup + task for 8 experts |
| 8.3 Expert prefetch | ✅ DONE | Full-stride to L3 (was 256B to L1) |
| 8.4 Output proj split | ✅ DONE | OMP outer loop for prefill (N>1) |

## Phase 9: KV Cache for GQA [NEXT — P0]
GQA layers (10 of 40) recompute full attention each decode step. As context grows, this becomes O(n²). Need KV cache:
- Allocate k_cache/v_cache per GQA layer
- On decode: append new K,V → compute attn against full cache
- On prefill: write all positions to cache
- Expected: decode attn time constant O(n_cache) not O(n_ctx)

## Phase 10: Cos-sim 1:1 Parity [P0]
Layer-by-layer comparison shows:
- L0 (SSM): cos=0.86 — first SSM layer diverges
- L1 (SSM): cos=0.75 — worsens
- L3 (GQA): cos=0.92 — GQA also diverges
- L6-L31: gradual 0.97→0.88 — cumulative quant noise
- L32-L39: sharp drop 0.88→0.46 — systematic

Root cause hypothesis: quantization path differences (bytropix quantized_matmul vs llama.cpp ggml_mul_mat) produce slightly different per-layer outputs, compounded by recurrence in SSM layers.

Fix: Dump intermediate values (Q, K, V, gate, beta, conv_out) from both and compare. Add SSM state dump to llama.cpp via the existing DUMP_LAYER_DIR infrastructure.

## Phase 11: SSM AVX2 Optimization [P2]
SSM layers: 0.8ms/layer × 30 layers = 24ms. Low priority vs GQA KV cache.

## Phase 12: NV64 RDRAM Ring Buffer [P1]
- ring_slot_t[64] with atomic head/tail
- Prefetch agent thread (graduated T2→T1→T0)
