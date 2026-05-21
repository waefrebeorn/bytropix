# Plan — May 21, 2026 (P1 Complete, P2 Active)

## Completed P1
1. ✅ MTP spec decode — gen_text_mtp working at 8.5 tok/s (4% acceptance from quantized head)
2. ✅ Vision pipeline — screenshot→encoder→mmproj→text→logits verified
   - 2 segfault bugs fixed in wubu_vision.c (n_patches_total cap, scores heap alloc)
   - 256×256 → 128 patches × 2048, no NaN, logit range [-10.8, 14.1]
   - Makefile test_vision_real target fixed with GPU_SUPPORT

## P2: Feature Cream
| Feature | Priority | Status |
|---------|----------|--------|
| GPU RMSNorm + SiLU + gated norm kernels | High | Not started |
| Chunked prefill (3-7x speedup) | High | Not started |
| RoPE extrapolation 4x | High | Not started |
| Sparse attention (NSA, DeepSeek V3.2) | High | Not started |
| GPU vision encoder kernels | High | 27 ViT all CPU → 63.7s for 256x256 |
| Sigmoid gating + load balancing (DeepSeekMoE) | Mid | Not started |
