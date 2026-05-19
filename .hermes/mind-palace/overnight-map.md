# Overnight Map — May 19, 2026 PM (Phase 16 Complete, GPU SSM Wired)

## Phase 15: GPU GQA Wiring ✅
Integrated into wubu_model_t. GPU=1 env var enables GPU GQA.

## Phase 16: GPU SSM Quantized Matmuls ✅
### Done
- `gpu_quant_matmul.cu/h` — Q5_K/Q6_K dequant+matmul kernels ✅
- Verified: GPU kernel cos-sim = **1.0 vs F32 dequant reference** ✅
- SSM quantized weights uploaded to GPU (692MB for 30 layers) ✅
- `wubu_model_gpu_ssm_project()` — GPU quantized qkv + gate projections ✅
- `wubu_ssm_forward()` — accepts optional gpu_qkv/gpu_z params to skip CPU matmuls ✅
- `wubu_model.c` — GPU SSM path added (parallel to GPU GQA path) ✅
- End-to-end: `GPU=1 ./gen_text_gpu "Hello" 8` runs with hybrid SSM ✅

### Known Issues
- GPU SSM is slower than CPU (2.3 vs 6.4 tok/s) — bottleneck is CPU SSM pipeline (conv, norm, recurrence), not GPU matmuls
- Text output differs from CPU-only due to Q8_K quantization noise in CPU quantized_matmul — GPU uses F32 dequant which is more accurate
- `wubu_model_gpu_ssm_project` does alloc/free per call — should use pre-allocated persistent buffers
- 6 stream syncs per SSM layer × 30 layers = 180 syncs/token

## Verdict
GPU kernel IS CORRECT. Cos-sim 0.90 vs CPU quantized_matmul was expected — CPU uses Q8_K quantized x (vec_dot), GPU uses raw F32 x. GPU matches F32 dequant reference exactly (cos-sim 1.0, max_err 0.0).

## Next: Phase 17 — GPU MoE
IQ2_XXS/IQ3_XXS quantized GPU kernels for MoE expert weights.

## Files Changed
- `src/gpu_quant_matmul.cu` — F16 denormal fix, extern "C" for C linkage
- `include/gpu_quant_matmul.h` — fixed header (cuda_runtime.h include, extern "C" guard)
- `src/wubu_ssm.c` — gpu_qkv/gpu_z optional params to skip CPU matmuls
- `include/wubu_ssm.h` — updated function signature
- `src/wubu_model.c` — GPU SSM path with GPU_SUPPORT guard
- `include/wubu_model.h` — wubu_model_gpu_ssm_project declaration
