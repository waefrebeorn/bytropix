# State — May 19, 2026 PM — Phase 15 Complete (GPU GQA Wired) ✅

## FINAL STATUS
- **Cos-sim vs llama.cpp: 0.9967** — 1:1 PARITY ACHIEVED ✅
- **Decode: 8.8 tok/s CPU** (16 threads, AVX2, fused Q8_K, tiled GQA) ✅
- **Decode: 3.5 tok/s GPU** (GPU GQA + CPU SSM/MoE + GPU out) ⚠️
- **Phase 15: GPU GQA wiring** — integrated into wubu_model_t ✅
  - F32 dequantized weights (10 layers × 1.04GB VRAM) on GPU
  - Persistent GPU KV cache (262144 positions, ~10MB/layer)
  - Per-layer chunked attention with RoPE + gate + output proj
  - Output verified matching CPU (tokenization variance only)
- **GPU GQA kernels written** (tile-streaming, softmax reduction, V weighted sum) ✅
- **MTP logit correction EMA** implemented (running correction draft[0]/draft[1]) 🔶
- **KV cache: 262144 F16 positions** (5GB, fits 16GB laptop) ✅
- **All 7 quant types verified** vs F32 SGEMM ✅
- **8 bugs fixed**, including the Q6_K one-character loop count ✅

## Hot Components
| Component | Time | % | Next Step |
|-----------|:----:|:-:|-----------|
| MoE (40 layers × 1.2ms) | 48ms | 48% | GPU MoE kernel (Phase 17) |
| SSM + GQA (40 layers × 1.0ms) | 40ms | 40% | GPU SSM matmuls (Phase 16) |
| Output proj (Q4_K, 2048×248320) | 10ms | 10% | Already GPU-accelerated |
| GPU GQA sync overhead | ~3ms | 3% | Reduce syncs, pipeline layers |
| Other | 2ms | 2% | Tokenizer optimization |

## Cold Gaps
| Prio | Gap | Status | Phase |
|------|-----|--------|-------|
| P1 | Wire GPU GQA attention into inference loop | ✅ Done! | 15 |
| P2 | GPU SSM matmuls (Q5_K dequant+matmul) | 🟡 Not started | 16 |
| P3 | GPU MoE expert compute (IQ2_XXS/IQ3_XXS) | 🟡 Not started | 17 |
| P4 | GPU MTP pipeline (overlap draft+verify) | 🟡 Not started | 18 |
| P5 | End-to-end GPU inference (~66 tok/s ceiling) | 🟡 Not started | 19 |
| ❌ | MTP verify at IQ2_M | ❌ 18% acceptance — EMA correction might help | — |
| ❌ | 256k context GQA on CPU | ❌ Scaled attention needed (NSA, streaming, or tiered) | — |
