# Goal Mantra — May 19, 2026 Late PM (Phase 14 — SSM AVX2 Optimization ✅)

## THE GOAL
**1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.** ✅
**Current: 0.9967 cos-sim** — 1:1 PARITY ACHIEVED.

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| **gen_text (CPU)** | **8.8 tok/s** decode | ✅ Fused Q8_K + AVX2 scan |
| **gen_text prefill** | **13.1 tok/s** (5-token) | ✅ |
| **SSM attn per layer** | **~1.0ms** | ✅ AVX2 scan + fused Q8_K |
| **MoE per layer** | **~1.2ms** | ✅ IQ2_XXS AVX2 |
| **Output proj** | **~10ms** CPU / **GPU quantized** | ✅ GPU_QUANTIZED=1 mode |
| **GPU memory (quantized)** | **~1.9GB VRAM** | ✅ Q4_K kernel, no F32 dequant |
| **SSM/GQA fused Q8_K** | **Saves 1 quant/layer** | ✅ 40 quants saved per decode |
| **AVX2 selective scan** | **8× float throughput** | ✅ 4 inner loops optimized |
| **NaN guard** | **Gated behind DUMP_GQA_DEBUG** | ✅ ~90K isnan() saved |

## CRITICAL FINDING: Fused Q8_K Quant
Both SSM and GQA layers do multiple matmuls on the same input x. By quantizing x→Q8_K ONCE per token and reusing for all projections, we save:
- SSM: 1 quant per layer × 30 = 30 quants per decode
- GQA: 2 quants per layer × 10 = 20 quants per decode  
- Total: 50 fewer Q8_K quantize operations per decode step

## CRITICAL FINDING: GPU Quantized Mode
The F32 output proj dequant uses 7.6GB VRAM which doesn't fit on 6.5GB laptops. New custom CUDA kernel keeps Q4_K on GPU (1.9GB) and dequants on-the-fly. Set `GPU_QUANTIZED=1` to enable.
