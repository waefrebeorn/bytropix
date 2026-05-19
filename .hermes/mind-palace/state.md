# State — May 19, 2026 Late PM (Phase 14 — SSM AVX2 Optimization ✅)

## REAL STATUS
- **Cos-sim vs llama.cpp: 0.9967** ✅ (AVX2 scan verified: max diff 1.85e-3)
- **KV cache: 262144 positions, F16 format** ✅
- **gen_text decode: 8.8 tok/s CPU** (16 threads, AVX2, fused Q8_K quant) ✅
- **gen_text prefill: 13.1 tok/s** (5-token prompt) ✅
- **gen_text_gpu: Q4_K quantized GPU mode** (GPU_QUANTIZED=1, uses ~1.9GB VRAM) ✅
- **SSM per-layer: ~1.0ms** ✅ (AVX2 scan + fused Q8_K quant)

## What's Hot
| Component | Time/token | % | Priority |
|-----------|-----------|---|---|
| MoE (40 layers × ~1.2ms) | ~48ms | 48% | P2 — IQ2_XXS/IQ3_XXS AVX2 done |
| SSM/GQA (40 layers × ~1.0ms) | ~40ms | 40% | P3 — AVX2 scan + fused Q8_K done |
| Output proj (2048×248320) | ~10ms | 10% | P1 — Q4_K AVX2, GPU quantized mode |
| **Total compute** | **~98ms** | **100%** | **Ceiling: ~10.2 tok/s** |

## Critical Findings
1. **Fused Q8_K quantization**: SSM attn_qkv + attn_gate now share one Q8_K quant per token. GQA Q+K+V also share. Saves ~1 quant per layer.
2. **AVX2 selective scan**: 4 inner loops (state decay, h@k, state update, h@q) now use AVX2 intrinsics. Each processes 8 floats per instruction vs scalar.
3. **GPU quantized mode**: New GPU_QUANTIZED=1 mode keeps output weight Q4_K on GPU (~1.9GB VRAM vs 7.6GB F32). Custom CUDA kernel dequants on-the-fly.
4. **NaN guard gated**: Only checks when DUMP_GQA_DEBUG is set — saves ~90K isnan() calls per decode.

## Cold Gaps
| Prio | Gap | Status |
|------|-----|--------|
| P2 | GPU output proj (cuBLAS) | Done — Q4_K quantized mode added |
| P3 | SSM AVX2 optimization | ✅ SSE->AVX2 in 4 scan loops |
| P3 | Fused Q8_K quant (SSM + GQA) | ✅ |
| P3 | NaN guard optimization | ✅ Gated behind env var |
| P4 | 256k context speed | Future — GQA attention O(n²) per layer |
| P5 | MTP speculative decode at UD-IQ2_M | ❌ Blocked by quantization |
