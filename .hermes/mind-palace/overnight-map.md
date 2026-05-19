# Overnight Map — May 19, 2026 PM (Phase 16 Done, Phase 17 Partial)

## Phase 16: GPU SSM Quantized Matmuls ✅
- Q5_K/Q6_K dequant+matmul GPU kernels: cos-sim 1.0 vs F32 ref ✅
- Wired into forward pass: `GPU=1 ./gen_text_gpu` uses hybrid SSM ✅
- Optimized: pre-alloc buffers, reduced syncs (6→2), decode 2.3→3.4 tok/s ✅

## Phase 17: GPU MoE (In Progress)
### Done
- GPU IQ2_XXS dequant+matmul kernel (__constant__ memory for grid tables)
- Single-expert kernel with fused gate→SiLU→up→down in shared memory
- Hotfix: misaligned uint32 access → byte-by-byte load
- wubu_gpu_moe_init() → uploads tables to constant memory
- wubu_gpu_moe_forward_experts() → per-expert kernel launches
- wubu_model_gpu_moe_experts() → wrapper for CPU MoE integration
- Kernel verified: produces non-zero output, no CUDA errors

### Remaining
- Wire into forward pass (replace CPU expert matmuls in wubu_moe_forward)
- Handle IQ3_XXS for shared expert weights

## Files Added/Changed
- `src/gpu_moe_kernel.cu` — GPU MoE expert kernel (NEW, 195 lines)
- `include/gpu_moe_kernel.h` — header (NEW)
- `src/iq2xxs_grid_data.inc` — grid tables (NEW)
- `include/wubu_moe.h` — added gpu_ctx field
- `include/wubu_model.h` — added wubu_model_gpu_moe_experts decl
- `src/wubu_model_gpu.cu` — MoE wrapper, init call
- `Makefile` — gpu_moe_kernel.o build rule + link
