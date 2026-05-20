# State — Phase 28e: Q6_K DEQUANT BUG FIXED! GPU SSM still diverges from CPU

**bytropix: text + vision multi-modal inference engine**
**GPU SSM C==1 decode: Q6_K dequant fix applied, GPU output anti-correlated to CPU (cos-sim -0.66)**

## Q6_K Dequant Bug Fixed (May 20)

**Root cause of 365x constant offset:** GPU Q6_K matmul kernel subtracted `32.0` instead of `d * sc * 32`:

```c
// BUG (was):
sum += (double)x[idx0] * ((double)d * sc[is+0] * v6 - 32.0);
// FIX:
sum += (double)x[idx0] * (double)d * (double)sc[is+0] * (double)(v6 - 32);
```

**Effect:** With gated norm output RMS ~1 and non-zero mean, SSM output was CONSTANT ~365 across ALL 2048 output elements. Fixed to std ~0.036.

## Remaining: GPU vs CPU diverges at cos-sim -0.66

- CPU path (FORCE_CPU_SSM): cos-sim 0.994 vs llama ✅
- GPU path (forward_full): cos-sim -0.656 vs CPU path ❌

### Verified Correct (matching CPU path):
- Q5_K quant matmul (column-major) ✅
- Q6_K quant matmul (column-major) ✅ (FIXED)
- L2 normalization (L2 norm = 1.0 per head) ✅
- q_scale = 1/sqrt(D_STATE) ✅
- Gated norm (same formula, same norm_weight) ✅
- Beta/alpha kernel (generates sigmoid/softplus values) ✅
- Conv/silu/split kernel ✅

### Suspect Areas:
1. **SSM recurrence state** — initialized to zero on GPU, may not persist correctly between layers/steps
2. **Conv state management** — shifted by fused conv kernel, initial state may not match CPU
3. **GPU/CPU state copy** — ssm_states vs d_ssm_state may diverge after forward_full returns
4. **Fused conv/silu kernel** — may handle conv_state shift differently from CPU

## Vision Encoder Status
- 384 LoC in `src/wubu_vision.c` — 27-layer 3D ViT with mmproj
- 111 LoC in `include/wubu_vision.h` — full API
- 106 LoC in `tools/test_vision_real.c` — E2E test + text pipeline
- **Status:** Written but NOT YET BUILT/TESTED in current session

## Quick Reference
| Item | Path | Notes |
|------|------|-------|
| Q6_K dequant fix | `gpu_quant_matmul.cu:114` | Applied in c07cf14 |
| GPU recurrence | `gpu_ssm_recurrence.cu:109` | Output with q_scale |
| Gated norm kernel | `cuda_kernels.cu:248` | |
| Beta/alpha fused | `cuda_kernels.cu:2723` | |
| Conv/silu/split | `cuda_kernels.cu:2757` | |
| CPU L2 norm (ref) | `wubu_ssm.c:176` | |
| CPU q_scale | `wubu_ssm.c:530-534` | |
| Vision encoder | `src/wubu_vision.c` | 384 LoC, 3D ViT |
| Vision test E2E | `tools/test_vision_real.c` | Vision→text pipeline |
| mmproj GGUF | `/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf` | Vision projection |
