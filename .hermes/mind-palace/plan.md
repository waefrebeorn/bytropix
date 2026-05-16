# bytropix Roadmap — llama.cpp Alignment Plan

Study done: llama/ dir has key reference files

## Phase 0: Correctness Fixes (NOW)

- [ ] **Shared expert gate**: `wubu_moe.c` line 72 sets `ffn_gate_inp_shexp = NULL`. Qwen3.5 shared expert needs sigmoid gate per-token. Load tensor, apply sigmoid, multiply into shared expert output before residual. Reference: `qwen35moe.cpp` lines 409-420.
- [ ] **Verify top-1 token "Doug" vs llama "Here"**: Run GPU SSM only (MOE=0) and compare layer-by-layer hidden states against CPU reference. If GPU TF32 causes divergence, make TF32 opt-in via env var.

## Phase 1: Inference Speed (Multi-Token)

### P1a — Chunked DeltaNet Scan (3× prefill speedup)
- Implement `build_delta_net_chunking` from `delta-net-base.cpp` lines 60-288
- CHUNK_SIZE=64, O(T·d²/CS + CS²·d) vs O(T·d²)
- CPU path first, then GPU
- Reference: `llama/src/models/delta-net-base.cpp`

### P1b — Fused Gate+Up MoE Projection (2× MoE matmul)
- Qwen3.5 stores `ffn_gate_up_exps` as fused [D_MODEL, 2*D_FF] weight
- Load once, matmul once, split output in half → gate_out, up_out
- Saves one large matmul per active expert per token
- Reference: `qwen35moe.cpp` line 393

### P1c — Single-Pass Top-K (minor MoE speedup)
- Replace O(E·K) repeated max with single O(E) selection
- std::nth_element or max-heap if available, else simple single-pass

## Phase 2: GPU Optimization

### P2a — Warp-Level Parallel SSM Scan (GPU occupancy)
- Replace `ssm_parallel_scan_kernel` with warp-level approach from `ssm-scan.cu`
- Each warp handles `c_factor` state dimensions, `warp_reduce_sum` for output
- Avoids 128-register `h_row` array, improves occupancy
- Template kernel for compile-time unrolling
- Reference: `llama/ggml-cuda/ssm-scan.cu`

### P2b — Conv State Build as Device Kernel
- Replace host `for` loop with `cudaMemcpyAsync` per-batch
- Single kernel copies conv_state + qkv_all → conv_input
- Also kernel for conv_state update (last k-1 elements)

### P2c — Conv1d Kernel with Shared Memory Cache
- Load CONV_KERNEL × channel block into shared memory
- Avoids 4× global reads per output element for k=4 conv
- 32KB fits in 48KB shared memory on Blackwell

## Phase 3: Quantized Inference (Memory)

### P3a — On-the-Fly IQ2_XXS Dot Product
- Replace `gguf_read_tensor_f32` dequant-all-at-load with per-block dequant during matmul
- Start with IQ2_XXS (primary MoE quant), then IQ3_XXS, IQ4_XS
- C function: `dequant_and_dot(const uint8_t *blocks, const float *vec, int n_elems, float *out)`
- Enables running without pre-dequantizing 10.9GB → f32 (saves ~15GB RAM)

### P3b — K-Quant Support
- Add Q4_K, Q5_K, Q6_K dequant + dot-product kernels
- Required for attention weights (currently dequantized at load)

## Phase 4: Architecture Alignment

### P4a — Model Graph Approach
- Consider minimal graph: register ops → dependency analysis → execute
- Not full ggml, but a lightweight version for bytropix
- Enables operator fusion (silu+gate, l2_norm+recurrence)

### P4b — KV Cache Manager
- Replace flat arrays with indexed cache (seq→slot mapping)
- Support batched inference with different sequence lengths
- Auto-cache-memory management for 256K context

## Phase 5: Training

### P5a — Backward Checkpointing
- Replace full-intermediate save (16 mallocs/layer) with recomputation
- Save states every `checkpoint_chunk_size` timesteps
- Trade 2× compute for memory reduction from O(T·d²) to O(CS·d²)

## Reference Files in llama/
| File | What We Learn |
|------|--------------|
| `src/models/delta-net-base.cpp` | Chunked DeltaNet algorithm (P1a) |
| `src/models/qwen35moe.cpp` | Fused gate+up MoE, shared expert gate (P1b, Phase 0) |
| `src/models/qwen3moe.cpp` | MoE graph builder pattern |
| `ggml-cpu/ops_ssm_sections.cpp` | CPU SSM conv + scan reference |
| `ggml-cuda/ssm-scan.cu` | Warp-level parallel scan CUDA (P2a) |
| `ggml-cpu/quants.c` | Quantized dot-product kernels (P3a, P3b) |
| `ggml-common.h` | GGUF type IDs, dequant table constants |
