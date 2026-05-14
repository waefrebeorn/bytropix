# WuBuText AI — Project Overview (May 14 PM)

## Mission
Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry. Inference engines complete. Training pipeline blocked by GPU weight loading bug.

## Done ✅
| Component | Status | Tool |
|-----------|--------|------|
| GGUF reader (13 types) | ✅ | train_real loads 733 tensors |
| SSM forward (30 layers) | ✅ CPU/GPU | infer_poincare 2835 tok/s |
| GQA forward (10 layers) | ✅ CPU | infer_unified 40-layer |
| MoE forward (256 experts) | ✅ lazy dequant | infer_moe_lazy 0.35s (9×) |
| Vision encoder (27 layers) | ✅ GPU 217ms | infer_vision_gpu |
| KV cache design | ✅ max_diff=0 | test_kv_cache |
| TGT NaN fixes | ✅ committed | tgt_wrap everywhere |
| GQA backward | ✅ wired | all 40 layers get gradients |
| Tokenizer | ✅ CJK round-trip | test_tokenizer |
| CPU timing (7 tests) | ✅ | cpu_timing.h |

## Broken ⛔
| Component | Failure | Priority |
|-----------|---------|----------|
| GPU weight loading | bench_e2e all zeros | P0 |
| GPU training | train_gpu CE 69 vs 12.66 | P0 |
| Gradient training | train_backprop hangs | P1 |

## Constraints
- **English only** — no CJK in code/comments
- **Pure C + CUDA** — no Python core
- **Verify ALL claims** — run binary, paste output
