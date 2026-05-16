# bytropix Roadmap — llama.cpp Alignment Plan

## Phase 0: Correctness Fixes ✓ DONE
- [x] **Shared expert gate**: Loaded + sigmoid applied. Verified build clean.
- [x] **MoE contiguous dequant**: Fixed. Output Chinese→English.
- [x] **MOE=1 default**: Changed from 0.
- [x] **MAX_LAYERS=0 clamp**: Fixed.

## Phase 1: Inference Speed ✓ DONE
- [x] **P1a — Chunked DeltaNet**: Implemented. NOTE: chunked≠sequential for multi-token chunks (different linear system). Training-only. Not used in inference.
- [x] **P1b — Fused Gate+Up**: Use existing separate weights (Qwen3.6 stores gate/up separately, not fused).
- [x] **P1c — Single-Pass Top-K**: O(EK) worst-first queue, replaced O(EK²) bubble sort.

## Phase 2: GPU Optimization ✓ DONE
- [x] **P2a — Warp-Level SSM Scan**: Template kernel ssm_warp_scan_kernel<4,128> at 1024 thr/block.
- [x] **P2b — Conv State Device Kernel**: build_conv_input_kernel + update_conv_state_kernel.
- [x] **P2c — Conv1d Shared Memory**: Conv weights cached in __shared__ per block. 
- [x] **TF32 math mode**: cuBLAS with CUBLAS_TF32_TENSOR_OP_MATH.
- [x] **Block size 512**: Element-wise kernels.

## Phase 3: Quantized Inference ✓ DONE
- [x] **P3a — On-the-Fly IQ2_XXS Dot**: dequantize_iq2_xxs_block(), dot_row functions. 4/4 tests pass.
- [x] **P3b — K-Quant Support**: Q4_K, Q5_K, Q6_K all have raw_size + dequant functions. All 7 model types supported.

## Phase 4: Architecture Alignment (ON HOLD)
- [ ] **P4a — Model Graph**: Not needed until operator fusion required. Current sequential execution works.
- [ ] **P4b — KV Cache Manager**: Not needed until 256K context deployment.

## Phase 5: Training (FUTURE)
- [ ] **P5a — Backward Checkpointing**: Not started.

## Current Issue
- **"Doug" vs llama "Here"**: Root cause unknown. attn_output_gate already implemented. Possible: tokenizer BOS, embd_norm epsilon, or quantization noise at 2-3 bpw on 35B.
