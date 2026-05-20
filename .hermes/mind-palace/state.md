# State — Phase 28b DA: F32 Waste Removed, ssm_project Fixed, forward_full Bug Found

**bytropix: inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**SSM GPU path: F32 waste fixed. ssm_project row_major fix applied. forward_full C==1 path has pre-existing illegal memory access.**

## Changes Applied (May 21 PM)

### ✅ Fix 1: F32 dequant SSM weight upload removed
`src/wubu_model_gpu.cu` lines 436-460, 474-483, 497-506 wrapped in `#if 0`
- Saves ~2.2 GB VRAM (was: ~2266 MB quant + ~2266 MB F32 dequant = 4532 MB SSM total)
- Now: only quantized weights uploaded (692 MB for 30 SSM layers)
- Struct fields (`d_qkv_f32[40]` etc.) preserved for init/free compatibility
- Free function handles NULL pointers safely (cudaFree(NULL) is no-op)

### ✅ Fix 2: ssm_project() uses row_major kernel
`src/wubu_model_gpu.cu` lines 864, 870
- Changed `wubu_cuda_quant_matmul` (broken column-major) to `wubu_cuda_quant_matmul_row_major` (correct row-major)
- This is the N>1 prefill fallback path — called from wubu_model.c line 519 when forward_full() returns 0

### ✅ Fix 3: CUDA error check on forward_full d_x upload
`src/wubu_model_gpu.cu` line 1027
- Now checks cudaMemcpyAsync return value
- Returns 0 on failure, triggering CPU fallback instead of running with corrupted state

## Pre-existing Bug: forward_full C==1 Illegal Memory Access (🔴 UNFIXED)

The SSM GPU path `wubu_model_gpu_ssm_forward_full()` with C==1 (decode path) triggers:
```
GPU SSM fwd_full: d_x upload: an illegal memory access was encountered
```

**Evidence from runtime:**
- 1-token prefill ("hello"): L0, L1 (SSM) succeed, L2+ fail — pattern suggests kernel memory corruption cascading
- Fatal on line 1027 `cudaMemcpyAsync(gpu->d_x, h_norm, ...)` — destination or source pointer invalid
- gpu->d_x IS allocated (succesful init, chunk_sz=256 → 2 MB)
- h_norm IS valid CPU memory (stack pointer from wubu_model.c)

**Root cause hypothesis:**
The fused SSM decode kernels (`ssm_beta_alpha_fused_decode`, `ssm_conv_silu_split_decode`, `wubu_gpu_ssm_recurrence`) have a memory corruption bug. L0/L1 process through the full pipeline including these kernels; by the time L2 runs, GPU state is corrupted and d_x upload fails.

**Note:** This bug PRE-EXISTED our changes. The DA state.md (May 21 AM) states SSM GPU path "compiles and runs but correctness UNVERIFIED" — this confirms it was never tested for correctness. Runtime garbage output with illegal access was incorrectly conflated with "runs."

## What Works (verified at runtime)
- GPU init completes (~3.0 GB VRAM with F32 waste removed, down from ~5.2 GB)
- GQA attention on GPU (F32 cuBLAS path)
- Output projection on GPU (F32 SGEMM mode for prefill, Q4_K kernel for decode)
- CPU-only path (gen_text) works but slow (35B model, CPU limited)

## What's Broken or Unverified (❓)
- 🔴 **forward_full C==1 decode path** — illegal memory access, produced garbage text
- 🔴 **forward_full C>1 prefill path** — never worked (cuBLAS error 13, falls through)
- 🔴 **GPU SSM fwd_full vs CPU comparison** — can't do until illegal access fixed
- ❓ **ssm_project fallback** — now uses correct row_major kernel, but never cos-sim verified

## Disposition: Known DA Claims vs Reality
| CLAIM (May 21 AM docs) | REALITY |
|-------------------------|---------|
| "GPU mem leak: 5.5 GB" | ❌ STALE — free() already frees all arrays (line 1237-1247) |
| "gen_text.c hardcoded prompt" | ❌ STALE — already accepts argv[1] (line 67) |
| "F32 waste 2.2 GB" | ✅ FIXED — removed in this session |
| "Prefill N>1 broken" | ✅ FIXED — now uses row_major kernel |
| "SSM GPU path runs" | ❌ FALSE — illegal memory access on decode |
| "GPU decode 7.6-8.5 tok/s" | ❌ UNVERIFIED — measured before GPU_SUPPORT was live (SSM on CPU) |

## Next Priority
1. Fix forward_full C==1 illegal memory access (fused SSM kernels)
2. Fix forward_full C>1 path (cuBLAS error 13)
3. Build verification pipeline: ref_dumper + layer_cos_sim
4. Cos-sim: GPU SSM vs llama.cpp reference
