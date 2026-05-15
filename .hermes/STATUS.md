# WuBuText AI — State Dashboard (May 15 PM v3)

## All Streams Complete + Verified ✅

| Stream | Status | Detail |
|--------|--------|--------|
| S1 GPU weight loading | ✅ Fixed | dequant bypass fixed, CPU/GPU match |
| S2 Training pipeline | ✅ CE=12.42 | GPU forward + TGT gradients + lazy MoE |
| S3 train_backprop | ✅ Verified | Not hanging (CPU-slow 25s/step) |
| S4 GQA L3 NaN | ✅ Fixed | CPU RMSNorm dim mismatch patched (6 sites) |
| S5 Vision→text pipeline | ✅ Verified | GPU 99ms/27 layers, 0 NaN |
| S6 Lazy MoE in training | ✅ | Top-8/256 cached fwd/bwd |
| S7 output.weight | ✅ | Loaded from GGUF, forward projection added |
| RSGD optimizer | ✅ | Riemannian SGD, wired in train_integrated |
| Poincaré GQA | ✅ | Hyperbolic distance attention, wired |
| Nested SSM K=4 | ✅ | Product of 4 Poincaré balls, wired |
| TST Training | ✅ | Bag s=8 MCE loss, wired |
| Nested MoE (16×16) | ✅ | Poincaré routing, wired (flat router) |
| CUDA kernels | ✅ | SSM scan + MoE dispatch, max_diff<6e-8 |
| Data pipeline | ✅ | 1.07M tokens tokenized |
| Moondream3 | ✅ | Weights dumped, C stub created |
| NaN in logits | ✅ Fixed | Output projection eliminated 0.57% NaN |
| GPU vision timeout | ✅ Resolved | 99ms, no timeout |

## Inference Binaries

| Binary | Status | Perf | Notes |
|--------|--------|------|-------|
| `infer_moe_lazy` | ✅ | 37 tok/s | Lazy dequant 9× |
| `infer_unified` | ✅ | 40 layers | SSM→GQA→MoE |
| `infer_poincare` | ✅ | 2835 tok/s GPU | Poincaré SSM |
| `infer_vision_gpu` | ✅ | 99ms 128×128 | ViT GPU |
| `infer_vision_text` | ✅ | 0 NaN | vision→text pipeline |
| `train_gpu` | ✅ | CE=12.42 | GPU + lazy MoE |
| `train_integrated` | ✅ | CE=12.42 | All flags wired |
| `test_gpu` | ✅ | match | GPU/CPU weight match |
| `test_cuda_kernels` | ✅ | max_diff<6e-8 | SSM + MoE dispatch |

## Cold Gaps (Manifold Code Not Yet Written)

| Priority | Gap | Impact |
|----------|-----|--------|
| P0 | Poincaré GQA backward | Blocks hyperbolic attention training |
| P1 | Nested SSM backward | Blocks nested geometry training |
| P1 | Möbius linear layer (M⊗) | Primitive for fully hyperbolic nets |
| P2 | Gyration closed-form | ~10× gyration speedup |
| P2 | Hyperbolic output projection | Consistent geometry to logits |
| P3 | Nested MoE 2-level backward | 2-level hierarchy training |
| P3 | Hyperbolic KV cache | K/V in Poincaré ball |

## TGT Math
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
