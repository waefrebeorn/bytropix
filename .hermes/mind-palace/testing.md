# WuBuText AI — Testing Protocol (May 15)

## All 7 Original Streams Complete ✅

| Test Binary | Command | Status |
|------------|---------|--------|
| `infer_moe_lazy` | `./infer_moe_lazy [model] [layer] [T]` | ✅ Dequant < 1s, output match |
| `infer_unified` | `./infer_unified [model] [T]` | ✅ 40-layer forward, no segfault |
| `test_kv_cache` | `./test_kv_cache [model]` | ✅ max_diff=0.00 |
| `infer_vision_gpu` | `./infer_vision_gpu [model] [image]` | ✅ 217ms 256×256 |
| `infer_poincare` | `./infer_poincare` | ✅ > 2000 tok/s |
| `test_moe` | `./test_moe` | ✅ range [-0.028, 0.031], NaN=0 |
| `train_real` | `./train_real [model] [corpus]` | ✅ CE ~12.66 |
| `train_gpu` | `./train_gpu` | ✅ CE ~12.42, lazy MoE working |
| `train_backprop` | `./train_backprop` | ✅ Runs (CPU-slow 25s/step) |
| `bench_e2e` | `PATH=... ./bench_e2e` | ✅ GPU weights fixed |
| `infer_vision_text` | `./infer_vision_text [model] [mmproj]` | ✅ Vision→text, 0 NaN |

## Remaining Test Gaps

| Feature | Test Needed | Priority |
|---------|-------------|----------|
| Poincaré GQA | Compare output vs standard GQA | P2 |
| RSGD optimizer | Verify no divergence from AdamW baseline | P2 |
| GPU vision pipeline | Speed test (target < 1s full pipeline) | P1 |
| Data pipeline | Tokenize corpus → verify round-trip | P2 |
| Nested SSM | Verify curvature routing + no NaN | P3 |
