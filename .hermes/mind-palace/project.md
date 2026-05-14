# WuBuText AI — Project Overview (May 15)

## Mission
Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry.
All 7 original streams complete. Working on math games + manifold + optimizations.

## Done ✅ (May 14-15)

| Component | Status | Detail |
|-----------|--------|--------|
| GGUF reader (13 types) | ✅ | 733 tensors loaded |
| SSM forward (30 layers) | ✅ CPU/GPU | Poincaré 2835 tok/s |
| GQA forward (10 layers) | ✅ CPU | 40-layer unified |
| MoE forward (256 experts) | ✅ lazy dequant | top-8/256, 9× speedup |
| Vision encoder (27 layers) | ✅ GPU 217ms | Qwen 3D ViT |
| KV cache | ✅ max_diff=0 | 1GB/layer @ 256K |
| TGT NaN fixes | ✅ | tgt_wrap everywhere |
| GPU weight loading | ✅ Fixed | gguf_reader dequant path |
| Training pipeline | ✅ CE=12.42 | lazy MoE + TGT gradients |
| train_backprop | ✅ Verified | Not hanging (CPU-slow) |
| Vision→text pipeline | ✅ Real screenshot | 128 tokens, 0 NaN |
| Lazy MoE in training | ✅ | cached fwd/bwd |

## Remaining Work

| Area | Items | Priority |
|------|-------|----------|
| Bugs | Model logit NaN (~0.5%), CPU RMSNorm dim | P0 |
| GPU vision | Wire cuda_vision.cu into pipeline | P1 |
| RSGD | Riemannian SGD for Poincaré params | P1 |
| Poincaré GQA | Hyperbolic distance attention | P2 |
| Data pipeline | Tokenize corpus → binary | P2 |
| Nested SSM | K curvatures product of balls | P3 |
| Nested MoE | Poincaré distance + hierarchy | P3 |
| TST | Token Superposition Training | P3 |
| CUDA kernels | SSM scan, MoE dispatch | P4 |
| Moondream3 | Weight dump + C port | P4 |

## Constraints
- **English only** — no CJK in code/comments
- **Pure C + CUDA** — no Python core
- **Verify ALL claims** — run binary, paste output
