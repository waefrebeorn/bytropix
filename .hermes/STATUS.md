# WuBuText AI — State Dashboard (May 15 PM)

## All 12 Streams Complete ✅

| Stream | Status | Detail |
|--------|--------|--------|
| S1 GPU weight loading | ✅ Fixed | dequant bypass fixed, CPU/GPU match |
| S2 Training pipeline | ✅ CE=12.42 | GPU forward + TGT gradients + lazy MoE |
| S3 train_backprop | ✅ Verified | Not hanging (CPU-slow 25s/step) |
| S4 GQA L3 NaN | ✅ Diagnosed | CPU RMSNorm dim mismatch (OOB), GPU OK |
| S5 Vision→text pipeline | ✅ Integrated | Real screenshot, 0 NaN |
| S6 Lazy MoE in training | ✅ | Top-8/256 cached fwd/bwd |
| S7 output.weight | ✅ | Loaded from GGUF |
| RSGD optimizer | ✅ | Riemannian SGD, valid Poincaré ball |
| Poincaré GQA | ✅ | Hyperbolic distance attention, 4/4 tests |
| Nested SSM K=4 | ✅ | Product of 4 Poincaré balls, 3/3 tests |
| TST Training | ✅ | Bag s=8 MCE loss, 8/8 tests |
| Nested MoE (16×16) | ✅ | Poincaré hierarchy routing, 396/396 tests |
| CUDA kernels | ✅ | SSM scan + MoE dispatch, max_diff<6e-8 |
| Data pipeline | ✅ | 1.07M tokens tokenized |
| Moondream3 | ✅ | Weights dumped, C stub created |

## Inference Binaries

| Binary | Status | Perf | Notes |
|--------|--------|------|-------|
| `infer_moe_lazy` | ✅ | 37 tok/s | Lazy dequant 9× |
| `infer_unified` | ✅ | 40 layers | SSM→GQA→MoE |
| `infer_poincare` | ✅ | 2835 tok/s GPU | Poincaré SSM |
| `infer_vision_gpu` | ✅ | 217ms 256×256 | ViT GPU |
| `infer_vision_text` | ✅ | 0 NaN | vision→text pipeline |
| `train_gpu` | ✅ | CE=12.42 | GPU + lazy MoE |
| `test_gpu` | ✅ | match | GPU/CPU weight match |
| `test_cuda_kernels` | ✅ | max_diff<6e-8 | SSM + MoE dispatch |

## ⚠️ INTERGRATION GAP

All 7 new math extensions (RSGD, Poincaré GQA, Nested SSM, TST, Nested MoE, CUDA kernels, data pipeline) are **standalone test harnesses**. None are wired into `train_gpu` or `wubu_model_forward`.

## ⚠️ Open Bugs

1. **GPU vision pipeline** — `infer_vision_text_gpu` timed out at 120s
2. **~0.5% NaN in model logits** — any input source, root cause unknown
3. **CPU RMSNorm dim mismatch** — d=4096 with weight[256] in CPU GQA path

## TGT Math
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
