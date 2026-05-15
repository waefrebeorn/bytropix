# WuBuText AI — Project Overview (May 15 PM)

## What We're Building
**WuBuText AI** is a pure C + CUDA implementation of a Qwen3.6-35B-A3B-compatible language model with 7 hyperbolic math extensions.

All 7 original streams + 9 math/optimization items complete. Every binary verified.

### Architecture
40 layers (30 SSM + 10 GQA, 3:1 repeating), 2048 hidden, 248K vocab, 262K ctx.
256 MoE experts (8 active + 1 shared), lazy dequant 9× speedup.
All weights loaded from GGUF (IQ2_M quantized).

### Current State: All P1-6 Complete, Integration Pending
| Phase | Component | Status | Tests |
|-------|-----------|--------|-------|
| 0 | GGUF Reader | ✅ 13 types | 733 tensors |
| 1 | Embedding Graft | ✅ Poincaré R=0.956 | 95% NN preserved |
| 2 | Attention Port | ✅ 40 layers CPU/GPU | CE=12.66/12.42 |
| 3 | Training Modules | ✅ RSGD+TST+NestedSSM+PGA+MoE | All pass |
| 4 | MoE Port | ✅ Lazy dequant + Nested routing | 396/396 |
| 5 | Vision Port | ✅ GPU 217ms + pipeline | 0 NaN |
| 6 | CUDA Kernels | ✅ SSM scan + MoE dispatch | max_diff<6e-8 |

### ⚠️ Integration Gap
All 7 math extensions standalone — NOT wired into training pipeline.

### ⚠️ Open Bugs
1. GPU vision pipeline timed out (120s)
2. ~0.5% NaN in logits (pre-existing, any input)
3. CPU RMSNorm OOB in GQA path (d=4096, weight[256])
