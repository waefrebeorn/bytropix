# Overnight Map — Phase 28b: Fixes Applied, forward_full Bug Blocks Verification

**Active repo**: /home/wubu/bytropix/
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (qwen35moe arch)
**Current state**: F32 waste fixed ✅. ssm_project row_major fix applied ✅. forward_full C==1 illegal memory access 🔴 BLOCKER.

## Changes Applied This Session
1. `#if 0` wrapped F32 dequant upload blocks (save ~2.2 GB VRAM) — `src/wubu_model_gpu.cu`
2. Changed ssm_project() to use row_major kernel (fixes N>1 fallback) — `src/wubu_model_gpu.cu`
3. Added CUDA error check on forward_full d_x upload — `src/wubu_model_gpu.cu`
4. Updated all mind-palace docs with honest DA assessment

## Pre-existing Bug: forward_full C==1 decode path
- Illegal memory access on EVERY SSM layer during decode
- L0/L1 appear to succeed, L2+ fail → GPU state corruption from fused kernels
- `ssm_beta_alpha_fused_decode` or `ssm_conv_silu_split_decode` likely culprit
- Runtime output: garbage text (Chinese/Korean/Arabic mix, repeating illegal access errors)

## Estimated VRAM after fixes
- GQA weights: 1,040 MB (unchanged)
- SSM weights (quantized only): 692 MB (down from ~4,532 MB — saved 3.8 GB)
- KV cache (Q4_0): 1,440 MB (unchanged)
- Output projection: 1,900 MB (unchanged)
- MoE + scratch: ~460 MB (unchanged)
- **Total: ~5.5 GB** → now **~3.4 GB** with F32 waste removed
- Headroom on 8GB GPU: ~4.6 GB for context expansion

## Next Session: Fix forward_full, Then Verify
### P0: Fix forward_full C==1 illegal memory access
- Likely in fused SSM decode kernels (cuda_kernels.cu lines 2723-2810)
- Use MARK debug pattern: insert `fprintf` at kernel boundaries, find which kernel corrupts state
- Check: output buffer sizes, d_ssm_qkv_out bounds, d_conv_state offsets, d_ssm_state allocation

### P1: Fix forward_full C>1 prefill path
- cuBLAS error 13 (CUBLAS_STATUS_EXECUTION_FAILED)
- Currently falls through to ssm_project fallback (now correct with row_major)
- Likely related to F32 cuBLAS matmul for beta/alpha in C>1 path

### P2: Build verification pipeline
- `make ref_dumper` — needs libllama.so linkage
- `DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf "prompt"` — generates reference layer dumps
- `DUMP_LAYER_DIR=/tmp/our ./gen_text "prompt"` — generates our layer dumps
- `tools/layer_cos_sim /tmp/ref /tmp/our 40` — per-layer cosine similarity

### Key Files
| File | Issue | Status |
|------|-------|--------|
| src/wubu_model_gpu.cu:436-460 | F32 qkv dequant upload | ✅ #if 0 |
| src/wubu_model_gpu.cu:474-483 | F32 gate dequant upload | ✅ #if 0 |
| src/wubu_model_gpu.cu:497-504 | F32 out dequant upload | ✅ #if 0 |
| src/wubu_model_gpu.cu:864,870 | ssm_project column-major | ✅ row_major fix |
| src/wubu_model_gpu.cu:1027 | forward_full d_x upload | ✅ error check |
| src/cuda_kernels.cu:2723-2810 | Fused SSM decode kernels | 🔴 illegal access |
| src/wubu_model_gpu.cu:1067-1074 | C>1 cuBLAS error 13 | 🔴 unfixed |
| tools/ref_dumper.cpp | Reference dumper | ❓ needs build |
