# WuBuText AI — State Dashboard (May 14 PM)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9× speedup) | Lazy dequant: only top-8/256 experts. Output match verified. |
| `infer_unified` | ✅ | 40 layers in 1 binary, per-layer timing | SSM→GQA→MoE chain with lazy MoE integration. |
| `test_kv_cache` | ✅ | max_diff=0.00 vs full recompute | KV cache: 1GB/layer @ 256K, 2.6× speedup at T=8. |
| `infer_vision` | ✅ | CPU: 825ms (64×64), ~35s (256×256) | 27-layer 3D ViT, OpenMP enabled. |
| `infer_vision_gpu` | ✅ | GPU: 65ms (64×64), 217ms (256×256) | 161× speedup, cuBLAS. |
| `infer_poincare` | ✅ | GPU: 2835 tok/s (B=1,T=4) | Poincaré SSM on GPU. |
| `test_256k` | ✅ | MoE router O(T) at 256K | 4.3k tok/s to 65K tokens. |
| `train_real` | ✅ | CE loss 12.66, 0.2 tok/s CPU | Correct CPU training path. |
| `test_moe` | ✅ | range [-0.028, 0.031], NaN=0 | 36.6 tok/s. |
| `bench_e2e` | ⛔ | All zeros output | GPU weight loading path broken. |
| `train_gpu` | ⛔ | CE loss 69 vs 12.66 | Same root cause as bench_e2e. |
| `train_backprop` | ⛔ | Hangs at model init | Unknown. |

## TGT NaN/Inf Fixes (committed fefd426)

| Location | Fix | Effect |
|----------|-----|--------|
| SSM state decay | `tgt_safe_expf` clamp [-80,80] | No exp overflow |
| SSM state matrix | `tgt_wrap` = fmod(x+π,2π)-π | State bounded to [-π,π] |
| GQA attention scores | `tgt_wrap` before softmax | No overflow |
| GQA Q/K/V | NaN→0 guard | No corrupted input propagation |
| SGD optimizer | TGT remainder replaces clip[-10,10] | Direction preserved, magnitude bounded |

## Priority Queue
P0 — Fix GPU weight loading (bench.c gpu_load_ssm_layer → zeros)
P1 — NaN is pre-existing GQA L3 (memory corruption hypothesis — MoE load overwrites GQA input)
P2 — Gradient training (train_backprop hang)
P3 — Vision→model integration
P4 — Update GPU training to use lazy MoE
