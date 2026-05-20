# Goal Mantra — Phase 28e: Q6_K Dequant Fixed, Fix GPU SSM Divergence

**Target:** Fix GPU SSM state management → cos-sim > 0.99 with CPU path.
**Current:** Q6_K dequant FIXED. GPU still anti-correlated (cos-sim -0.66). Vision encoder ported (384 LoC).

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| Q6_K dequant | ✅ FIXED | `d*sc*(v6-32)` vs `32.0` |
| GPU SSM vs CPU | ❌ cos-sim -0.66 | Anti-correlated output |
| CPU SSM vs llama | ✅ cos-sim 0.994 | FORCE_CPU_SSM verified |
| F32 waste removed | ✅ Saved ~2.2 GB | `#if 0` in a032a8f |
| Vision encoder | ✅ 384 LoC ported | 3D ViT + mmproj pipeline |
| CPU gen_text build | ❌ BROKEN | GPU symbols in wubu_model.o |
| Remote push | 🔴 8 behind | All critical GPU fixes local |

## BUILD
```bash
make gen_text_gpu       # GPU inference
GPU=1 MAX_CTX=4096 ./gen_text_gpu "The capital of France is" 20 40
FORCE_CPU_SSM=1 GPU=1 ./gen_text_gpu "test" 5 40  # CPU fallback
```

## P0: Fix GPU SSM divergence
1. Debug dump at each stage: GPU vs CPU intermediate values
2. Check recurrence state persistence (layers/steps)
3. Check conv state init + shifting
4. Fix → cos-sim > 0.99

## P1: Infrastructure
5. Fix CPU gen_text build (`#ifdef GPU_SUPPORT`)
6. Push 8 commits to remote
7. Re-verify full cos-sim

## P2: Vision integration
8. Build test_vision_real → verify output
9. Wire full vision→text multi-modal pipeline

## EVERY FIX: compile → run → checkpoint cos-sim → document
