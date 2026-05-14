# WuBuText AI — Overnight Navigation Map (May 15)

## Where We Are

**All 7 original streams complete.** S1-S7 committed and verified. Moving to math games, manifold, and optimizations.

### S1-S7 All Verified ✅
- GPU weight loading fixed (626c143)
- Training pipeline with TGT gradients + lazy MoE, CE=12.42
- train_backprop verified (not hanging, just CPU-slow)
- GQA L3 NaN: CPU dim mismatch, not memory corruption
- Vision→text pipeline: real screenshot, 0 NaN
- Lazy MoE in training: top-8/256, cached fwd/bwd
- output.weight loaded from GGUF

### Remaining Work

| Priority | Work | Status |
|----------|------|--------|
| P0 | Pre-existing NaN in model logits (~0.5%) | ❌ |
| P0 | CPU GQA RMSNorm dim mismatch (d=4096, weight[256]) | ❌ |
| P1 | Wire GPU vision (cuda_vision.cu 217ms) into pipeline | ❌ |
| P1 | RSGD optimizer for Poincaré params | ❌ |
| P2 | Poincaré GQA (hyperbolic distance attention) | ❌ |
| P2 | Data pipeline (corpus→token IDs .bin) | ❌ |
| P3 | Nested SSM (K curvatures product of balls) | ❌ |
| P3 | Nested MoE (Poincaré router + hierarchy) | ❌ |
| P3 | TST Token Superposition Training | ❌ |
| P4 | Moondream3 weight dump + C port | ❌ |
| P4 | CUDA kernels (SSM scan, MoE dispatch) | ❌ |

### Build Command
```bash
PATH="/usr/local/cuda/bin:$PATH" make <target>
```
