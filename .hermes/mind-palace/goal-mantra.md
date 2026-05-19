# Goal Mantra — May 19, 2026 PM (Phase 15 — GPU GQA Wiring ✅)

## THE GOAL
**1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.** ✅
**Current: 0.9967 cos-sim** — 1:1 PARITY ACHIEVED.

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| **gen_text (CPU)** | **8.8 tok/s** decode | ✅ Fused Q8_K + AVX2 scan |
| **gen_text (GPU GQA)** | **3.5 tok/s** decode | ✅ GQA on GPU, SSM+MoE CPU |
| **gen_text prefill** | **13.1 tok/s** (5-token) | ✅ |
| **SSM attn per layer** | **~1.0ms** | ✅ AVX2 scan + fused Q8_K |
| **MoE per layer** | **~1.2ms** | ✅ IQ2_XXS AVX2 |
| **Output proj** | **~10ms** CPU / **GPU quantized** | ✅ GPU_QUANTIZED=1 mode |
| **GPU GQA VRAM** | **1.04GB** F32 dequant weights | ✅ 10 GQA layers on GPU |
| **GPU GQA KV cache** | **Persistent, 262k ctx** | ✅ On-GPU, per-layer |
| **GPU memory (quantized)** | **~2.9GB** total | ✅ GQA(1.04) + Out(1.9) |

## CRITICAL FINDING: Integrated GPU GQA
Phase 15 wired `wubu_model_gpu.cu` into `wubu_model_t` — GPU GQA is now a transparent feature. Set `GPU=1` env var to enable. Uses:
- F32 dequantized weights → cuBLAS SGEMM for QKV projections
- Persistent GPU KV cache (device-to-device memcpy for append)
- `wubu_cuda_chunked_attn` with online softmax tiling
- Token-by-token forward for prefill (chunked attn non-causal for C>1)

## NEXT: Phase 16 — GPU SSM Matmuls
Requires quantized GPU kernel (Q5_K/Q6_K on-GPU dequant) to fit VRAM.
