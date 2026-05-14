# WuBuText AI — State Dashboard (May 15)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9×) | Lazy dequant: top-8/256 experts |
| `infer_unified` | ✅ | 40 layers in 1 binary | SSM→GQA→MoE chain |
| `test_kv_cache` | ✅ | max_diff=0.00 vs recompute | KV cache: 1GB/layer @ 256K |
| `infer_vision` | ✅ | CPU: 825ms (64×64) | 27-layer 3D ViT, OpenMP |
| `infer_vision_gpu` | ✅ | GPU: 217ms (256×256) | 161× speedup, cuBLAS |
| `infer_poincare` | ✅ | GPU: 2835 tok/s | Poincaré SSM on GPU |
| `test_256k` | ✅ | MoE router O(T) at 256K | 4.3k tok/s to 65K tokens |
| `train_real` | ✅ | CE loss 12.66 | Correct CPU training |
| `test_moe` | ✅ | range [-0.028, 0.031], NaN=0 | 36.6 tok/s |
| `train_gpu` | ✅ | CE=12.42 with lazy MoE | GPU forward + lazy MoE |
| `train_backprop` | ✅ | Runs (CPU-slow ~25s/step) | Not hanging |
| `bench_e2e` | ✅ | GPU weight loading fixed | Match verified |
| `infer_vision_text` | ✅ | Vision→text pipeline | Real screenshot, 0 NaN |

## TGT NaN/Inf Fixes (committed)

| Location | Fix | Effect |
|----------|-----|--------|
| SSM state decay | `tgt_safe_expf` clamp [-80,80] | No exp overflow |
| SSM state matrix | `tgt_wrap` = fmod(x+π,2π)-π | State bounded to [-π,π] |
| GQA attention scores | `tgt_wrap` before softmax | No overflow |
| GQA Q/K/V | NaN→0 guard | No corrupted propagation |
| SGD optimizer | TGT remainder replaces clip | Direction preserved |

## Remaining Work

| Area | Items |
|------|-------|
| Math games | RSGD, Poincaré GQA, nested SSM, nested MoE |
| Manifold | Moondream3, Poincaré distance routing |
| Optimizations | GPU vision pipeline, data pipeline, TST, CUDA kernels |
| Bugs | Model logit NaN (~0.5%), CPU RMSNorm dim mismatch |
