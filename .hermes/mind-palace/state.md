# State — Phase 28i: GPU MoE IQ3_XXS Fixed ✓

**bytropix: text + vision multi-modal inference engine**

## Achieved

### GPU MoE IQ3_XXS Support (commit pending)
Fixed GPU MoE to handle IQ3_XXS down weights (instead of hardcoding IQ2_XXS for all).
- Added `iq3_xxs_dot()` GPU function with grid lookup + scale/sign dequant
- Added `d_iq3xxs_grid` constant memory (256-entry uint32 grid)
- Modified `moe_expert_kernel` to dispatch dequant by type (IQ2_XXS=66B/block, IQ3_XXS=98B/block, IQ4_XS=136B/block)
- Fixed scratch/cache buffer sizes from 270KB to 557KB (IQ4_XS max per expert)
- Removed `(void)gate_type/up_type/down_type` suppression

Full GPU inference (GQA + Q5_K + MoE) now produces coherent output. ✅
Test: "Paris is" → "the capital of France and the most populous city in the European Union. The city is located on the river Seine..."
Decode: ~5.4 tok/s on RTX 5050 8GB.

### Q5_K F16 Denormal Bug (bf573b8)
GPU `fp16_to_fp32_dev()` flushed denormals to zero. 90% of Q5_K blocks affected. Fixed. ✅

### GQA Fused Q+Gate Interleaved Layout Bug (cdccde2)
GPU copy kernels assumed contiguous Q/gate layout but actual layout is per-head interleaved. Fixed. ✅

## Remaining

### GPU SSM State Sync
Prefill/decode state sync still diverges. GPU SSM matmul path uses Q5_K (works) but conv/norm are CPU-only.
Hybrid path works for decode but prefill needs CPU SSM.

### GPU Recurrence Kernel
Untested after all fixes. CPU SSM path works correctly (cos-sim 0.994 vs llama.cpp).

## Next Step
Verify full 256k context with GPU. Or profile GPU MoE vs CPU MoE performance.
