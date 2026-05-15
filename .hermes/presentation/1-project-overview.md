# WuBuText AI — Project Overview (May 15 PM v6)

## What We're Building
**WuBuText AI** — pure C + CUDA implementation of Qwen3.6-35B-A3B with 7 hyperbolic math extensions.
All phases complete. Integrated training pipeline at 11s/step.

### Architecture
40 layers (30 SSM Gated DeltaNet + 10 GQA, 3:1 repeating), 2048 hidden, 248K vocab, 262K ctx.
256 MoE experts (8 active + 1 shared), per-expert IQ2_XXS dequant (3.9ms/expert).

### All Phases Complete ✅

| Phase | Component | Status | Key Metric |
|-------|-----------|--------|------------|
| 0 | GGUF Reader | ✅ 13 types | 733 tensors |
| 1 | Embedding Graft | ✅ Poincaré R=0.956 | 95% NN preserved |
| 2 | Attention Port | ✅ 40 layers CPU/GPU | GPU verified |
| 3 | Training Loop | ✅ Integrated | **11s/step, 0 NaN** |
| 4 | MoE Port | ✅ Per-expert dequant | 177s→11s/step (16×) |
| 5 | Vision Port | ✅ GPU 99ms pipeline | 0 NaN |
| 6 | CUDA Optimization | ✅ SSM scan + MoE dispatch | max_diff<6e-8 |

### Key Fixes
- **NaN root cause**: gguf_raw_size(IQ2_XXS) was 72→66 bytes/block
- **MoE magnitude**: hidden max 5e9→13 (buggy strided extraction fixed)
- **Training speed**: full tensor dequant eliminated → per-expert extraction
- **Bugs all closed**: vision timeout, NaN in logits, RMSNorm OOB — all ✅
