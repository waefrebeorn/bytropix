# State — May 19, 2026 PM — Phase 16 Complete (GPU SSM Quant Matmuls) ✅

## FINAL STATUS
- **Cos-sim vs llama.cpp: 0.9967** — 1:1 PARITY ACHIEVED ✅
- **Decode: 8.8 tok/s CPU** (16 threads, AVX2, fused Q8_K, tiled GQA) ✅
- **Decode: 2.3 tok/s GPU** (GPU GQA + GPU SSM proj + CPU SSM rest + CPU MoE) ⚠️
  - GPU SSM currently slower than CPU due to per-call alloc/free + sync overhead
  - GPU path is MORE ACCURATE than CPU (F32 dequant vs Q8_K vec_dot)
- **Phase 16: GPU SSM quantized matmuls** — wired into forward pass ✅
  - Q5_K/Q6_K dequant+matmul GPU kernels: verified cos-sim=1.0 vs F32 ref ✅
  - SSM quantized weights on GPU (30 layers × 692MB)
  - wubu_ssm_forward() accepts gpu_qkv/gpu_z to skip CPU matmuls
  - Token-by-token GPU projections then CPU conv/norm/recurrence
- **Phase 15: GPU GQA wiring** — integrated into wubu_model_t ✅
  - F32 dequantized weights (10 layers × 1.04GB VRAM) on GPU
  - Persistent GPU KV cache (262144 positions, ~10MB/layer)
- **GPU GQA kernels written** (tile-streaming, softmax reduction, V weighted sum) ✅
- **All 7 quant types verified** vs F32 SGEMM ✅
- **9 bugs fixed**, including Q6_K one-character loop count and F16 denormal in GPU fp16 converter ✅

## Hot Components
| Component | Time | % | Next Step |
|-----------|:----:|:-:|-----------|
| MoE (40 layers × 1.2ms) | 48ms | 48% | GPU MoE kernel (Phase 17) |
| SSM + GQA (40 layers × 1.0ms) | 40ms | 40% | Already partially GPU — optimize Phase 16 |
| Output proj (Q4_K, 2048×248320) | 10ms | 10% | Already GPU-accelerated |
| GPU sync overhead | ~5ms | 5% | Pre-allocate buffers, batch projections |
| Other | 2ms | 2% | Tokenizer optimization |

## Cold Gaps
| Prio | Gap | Status | Phase |
|------|-----|--------|-------|
| P1 | Wire GPU GQA attention into inference loop | ✅ Done! | 15 |
| P2 | GPU SSM matmuls (Q5_K dequant+matmul) | ✅ Done — wired into forward | 16 |
| P3 | GPU MoE expert compute (IQ2_XXS/IQ3_XXS) | 🟡 Not started | 17 |
| P4 | GPU MTP pipeline (overlap draft+verify) | 🟡 Not started | 18 |
| P5 | End-to-end GPU inference (~66 tok/s ceiling) | 🟡 Not started | 19 |
