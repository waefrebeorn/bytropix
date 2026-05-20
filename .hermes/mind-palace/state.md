# State — Phase 28f: GPU Output Proj + State Sync FIXED

**bytropix: text + vision multi-modal inference engine**
**Phase 28f status: GPU output projection ✅, SSM state sync ✅, forward_full still diverges from CPU**

## Achieved (This Session)
1. **GPU output projection fix:** Q4_K dequant was reading wrong layout (used ceil(V/256) stride instead of ceil(D/256)). Now stores as [V][D] row-major with CUBLAS_OP_T, ld=D. All 248k logits non-zero. ✅
2. **SSM state sync fix:** Prefill via hybrid path updates CPU ssm_state/conv_state. Decode via forward_full or hybrid recurrence used GPU state initialized to zero. Added CPU→GPU sync after hybrid prefill and before decode. Added GPU→CPU sync after forward_full decode. ✅
3. **Hybrid decode path active:** forward_full disabled for decode. Hybrid (GPU recurrence via wubu_gpu_set_ssm_hybrid + CPU conv/norm/recurrence) active with state sync. Produces approximately correct text (~coherent). ⚠️

## Remaining: GPU Quant Matmul Precision
- Prefill uses hybrid: GPU wubu_cuda_quant_matmul (Q5_K, F32 x) + CPU conv/norm/recurrence
- Pure CPU uses: Q8_K re-quantized x + Q5_K quant matmul via quantized_matmul_from_q8
- Differences in qkv/gate values compound across 30 SSM layers → first decode token diverges
- Hybrid output is ~coherent but not token-identical to CPU

## P0: Fix forward_full GPU SSM divergence
- forward_full produces near-zero output on first call (conv_state all-zero)
- After state accumulation, forward_full output is non-zero but wrong compared to hybrid
- Suspect: GPU recurrence kernel (ssm_recurrence_kernel) or conv1d kernel has subtle bug
- Currently disabled; decode uses hybrid path only

## P1: Infrastructure
- CPU gen_text build still broken (GPU symbols without #ifdef)
- Push commits to remote (8 behind)

## P2: Vision integration
- Build test_vision_real → verify output
- Wire full vision→text multi-modal pipeline

## Build
```bash
make gen_text_gpu                        # GPU inference
GPU_BATCH=5 GPU=1 MAX_CTX=4096 ./gen_text_gpu "prompt" 20 40
```

## Key Files Modified
| File | Change |
|------|--------|
| `src/wubu_model_gpu.cu` | Added `wubu_gpu_sync_ssm_state_to_gpu/cpu` functions |
| `src/wubu_model.c` | Wired state sync after hybrid prefill + before decode |
| `include/wubu_model.h` | Declared sync functions |
| `src/gpu_output_proj.cu` | Fixed Q4_K dequant layout + cuBLAS call |
