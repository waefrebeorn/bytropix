# State — Phase 28c: GPU SSM PATH RUNS WITHOUT CRASHES

**bytropix: inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**GPU SSM C==1 decode path: ALL 30 layers process without errors. Output correctness UNVERIFIED.**

## Root Cause Found and Fixed: Catastrophic Init Bug

The pre-existing "illegal memory access" bug was NOT in the fused SSM kernels, but in the weight initialization.

**Bug:** `wubu_model_gpu_init()` had 6 `for (int i = 0; i < 40; i++)` NULL-assignment loops INSIDE the per-layer loop (lines 417-422). Every time a new SSM layer was initialized, it would NULL-out ALL 40 layers' small F32 weight pointers (ssm_beta, ssm_alpha, ssm_dt_bias, ssm_a, ssm_conv1d, ssm_norm).

**Effect:** Only the LAST SSM layer (L38) retained its small F32 weights. All other SSM layers had NULL pointers. When `ssm_beta_alpha_fused_decode` was called, it dereferenced NULL GPU pointers, causing "an illegal memory access was encountered."

**Fix:** Removed the per-iteration re-zero loops. The arrays are calloc'd (zero-initialized) at the beginning of wubu_model_gpu_init, so no explicit NULL init is needed per-iteration.

## What Changed This Session

### ✅ Fix 1: F32 dequant SSM weight upload removed
- Wrapped in `#if 0` — saves ~2.2 GB VRAM

### ✅ Fix 2: ssm_project() uses row_major kernel
- Changed column-major to row_major in prefill N>1 fallback

### ✅ Fix 3: SSM small F32 weight init bug fixed
- Removed per-iteration `for (int i = 0; i < 40; i++) NULL` loops
- ALL 30 SSM layers now have valid small F32 weights on GPU
- GPU SSM forward_full C==1 decode path runs without crashes

### ✅ Fix 4: CUDA error checks added
- After every kernel call in forward_full
- NULL pointer checks before beta/alpha kernel
- Faster debugging for future issues

## Current Status
### Works (verified at runtime)
- GPU init: ~3.0 GB VRAM (down from ~5.2 GB with F32 waste removed)
- GQA attention on GPU (F32 cuBLAS)
- Output projection on GPU (F32 SGEMM)
- **GPU SSM forward_full C==1 decode: ALL 30 layers pass without any CUDA error**
- Prefill fallback (ssm_project with row_major)

### Still Broken
- 🔴 **Model produces garbage output (`<-1>` tokens)** — SSM GPU compute is likely incorrect
- 🔴 **No cos-sim comparison done** — GPU vs CPU vs llama.cpp never compared
- 🔴 **Prefill N>1 forward_full bypassed** (C>1 cuBLAS error 13, uses ssm_project fallback)
- 🔴 **Prefill GPU batch truncation** (gpu_output_project_batch T=5 > max=1)

### DA Assessment
| Claim | Reality |
|-------|---------|
| "F32 waste 2.2 GB" | ✅ FIXED |
| "Mem leak 5.5 GB" | ❌ STALE (was already fixed) |
| "ssm_project column-major" | ✅ FIXED (now row_major) |
| "SSM GPU path runs" | ✅ NOW FIXED (was NULL init bug) |
| "SSM GPU path correct" | ❓ UNKNOWN — runs without crash, output is garbage |
| "GPU decode tok/s" | ❓ UNKNOWN — garbage output, not measurable |

## Next Priority
1. **Fix correctness** — SSM GPU compute produces wrong values. Compare GPU/CPU hidden states.
2. **Build verification pipeline** — ref_dumper + layer_cos_sim
3. **Fix prefill N>1** — cuBLAS error 13 in forward_full
4. **Fix prefill batch truncation** — increase g_max_batch in gpu_output_proj
5. **Profile after verification**
