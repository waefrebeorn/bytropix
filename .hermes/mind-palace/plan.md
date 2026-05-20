# Plan — Phase 28b DA: Fix F32 Waste, Verify Correctness, Then Profile

## 🔴 DA Blockers

### P0: Remove F32 dequant SSM weight upload
File: `src/wubu_model_gpu.cu` lines 436-460, 474-483, 497-504
- Uploads F32 dequant weights for ALL 3 SSM weight tensors (qkv/gate/out) = ~128 MB/layer × 30 = ~3.8 GB
- NEVER used: `forward_full()` uses quantized row_major kernel, `ssm_project()` uses broken column-major
- Fix: comment out or `#if 0` the F32 upload blocks, save 2.2 GB VRAM for context

### P0: Fix wubu_model_gpu_free() memory leak
File: `src/wubu_model_gpu.cu` in `wubu_model_gpu_free()`
- Missing free calls for: d_attn_qkv_q[40], d_attn_gate_q[40], d_ssm_out_q[40], d_qkv_f32[40], d_gate_f32[40], d_out_f32[40]
- Each is a raw uint8_t*/float* per layer — loop over layers and `wubu_cuda_free()`

### P0: Fix prefill N>1 fallback — use row_major kernel
File: `src/wubu_model_gpu.cu` lines 864, 870 in `wubu_model_gpu_ssm_project()`
- Currently calls `wubu_cuda_quant_matmul()` (broken column-major)
- Change to `wubu_cuda_quant_matmul_row_major()` (correct row-major)

### P0: Fix gen_text.c prompt for proper testing
File: `tools/gen_text.c` line 63
- Hardcoded `const char *prompt = "The meaning of life is"`  
- Change to read from argv[optind] or stdin
- This blocks ALL verification work

## 🟡 Verification Phase (after P0 fixes)
- [ ] Cos-sim: GPU SSM vs CPU SSM at single layer (layer 0)
- [ ] Cos-sim: full 30-layer SSM path vs CPU
- [ ] Cos-sim: full 40-layer inference vs llama.cpp reference
- [ ] Verify prompt tokenization matches llama.cpp

## 🟢 Profile Phase (after verification)
- [ ] CUDA events: measure SSM GPU time per layer
- [ ] Compare tok/s: GPU SSM vs CPU SSM baseline
- [ ] Profile MoE, GQA, output proj separately
- [ ] Identify true bottlenecks (was: MoE ~20-40ms guess)

## Next: 256k Context
- [ ] After correctness verified at 4K, test at 256K
- [ ] Cos-sim vs llama.cpp at 256K
- [ ] VRAM headroom check after removing F32 waste

