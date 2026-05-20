# State — Phase 28g: Q5_K Denormal Bug FIXED, GPU Quant Matmul Matches CPU

**bytropix: text + vision multi-modal inference engine**
**Phase 28g: GPU Q5_K quant matmul ✅ cos-sim 0.9999999933 vs CPU proper F32 reference**

## Achieved This Session

1. **🔴 ROOT CAUSE IDENTIFIED: F16 denormals flushed to zero in GPU kernel** — 90% of Q5_K blocks (5.7M/6.3M) in this model have denormal F16 d/dmin values. GPU `fp16_to_fp32_dev()` returned ±0.0 for all of them. CPU `f16_to_f32()` correctly normalizes to `(-1)^sign * mant * 2^(-24)`. ✅ FIXED
2. **GPU Q5_K quant matmul VERIFIED** — cos-sim 0.9999999933 vs proper CPU F32 dequant reference (was -0.51 before fix). The old test was masked because its inline `f2` reference also flushed denormals. ✅
3. **GPU row-major variant** already handles denormals via while-loop normalization — no fix needed there. ✅

## What's Still Broken (Pipeline Issues)

Despite the Q5_K fix, GPU inference output is still garbage ("从现在, 1, 1, 1, 1, 1, 1," for "Paris is the capital of"). This is caused by **separate pipeline issues**:
- `GPU SSM C>1 path not yet working (cuBLAS error 13)` — F32 SSM matmul path fails, falls back silently
- `GPU input RMS=0.000000` in decode phase — GPU decoder receives zero input
- F32 cuBLAS SSM path also broken (VRAM pressure on RTX 5050 8GB?)

These are not regressions from the Q5_K fix — the same pipeline was broken before.

## Quick Build
```bash
make gen_text_gpu
GPU_BATCH=5 GPU=1 MAX_CTX=4096 ./gen_text_gpu "Paris is the capital of" 20 40
```
