# WuBuText AI — Project Overview (May 15 PM)

## Mission
Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry.
All 7 original streams + 9 math/optimization items complete.

## Done ✅ (May 14-15 Sprint)

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
| Vision→text pipeline | ✅ Real screenshot | 128 tokens, 0 NaN |
| Lazy MoE in training | ✅ | cached fwd/bwd |
| **RSGD optimizer** | ✅ | Riemannian SGD, valid ball |
| **Poincaré GQA** | ✅ | Hyperbolic dist attention, 4/4 |
| **Nested SSM K=4** | ✅ | 4 Poincaré balls, 3/3 tests |
| **TST Training** | ✅ | Bag s=8 MCE, 8/8 tests |
| **Nested MoE (16×16)** | ✅ | Poincaré hierarchy, 396/396 |
| **CUDA kernels** | ✅ | SSM scan + MoE dispatch |
| **Data pipeline** | ✅ | 1.07M tokens |
| **Moondream3** | ✅ | weights dumped + C stub |

## ⚠️ Integration Gap
All math extensions are standalone — no wiring into train_gpu.

## ⚠️ Open Bugs
1. GPU vision pipeline timed out (120s)
2. ~0.5% NaN in logits (pre-existing)
3. CPU RMSNorm OOB in GQA path

## Constraints
- **English only** — no CJK in code/comments
- **Pure C + CUDA** — no Python core
- **Verify ALL claims** — run binary, paste output
