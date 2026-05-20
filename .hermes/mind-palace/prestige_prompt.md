# Prestige Prompt — May 21 DA (Phase 28 Audit)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M

## PHASE 28 DONE: GPU_SUPPORT Now Live — But Unverified
Fixed 3 bugs to make GPU_SUPPORT compile and run for the first time. However, the DA audit reveals the SSM GPU path has NEVER been verified against a ground truth reference. Key issues:

## DA CRITICAL FINDINGS
1. **F32 dequant SSM weights waste ~2.2 GB VRAM** — uploaded but never used in GPU SSM path. `forward_full()` uses quantized row_major kernel. These F32 arrays consume 3.8 GB across 30 layers and are only used by the dead-code `wubu_model_gpu_ssm_project()` which uses the BROKEN column-major quant_matmul. **Fix: remove F32 upload.**

2. **Prefill N>1 produces garbage** — `wubu_model_gpu_ssm_project()` uses `wubu_cuda_quant_matmul()` (broken column-major kernel). This IS called from the N>1 fallback path when `forward_full()` fails. Fix: switch to `wubu_cuda_quant_matmul_row_major()`.

3. **GPU memory leak: ~5.5 GB never freed** — `wubu_model_gpu_free()` never frees d_attn_qkv_q[40], d_attn_gate_q[40], d_ssm_out_q[40], or any of the F32 dequant weight arrays (d_qkv_f32, d_gate_f32, d_out_f32).

4. **SSM GPU path output NEVER verified** — cos-sim vs CPU path: zero. The Phase 26 fused kernels were verified in isolation but never in the full inference pipeline. The row_major quant matmul kernel was compared only vs F32 dequant reference, never vs the CPU quantized_matmul path.

5. **gen_text.c blocks all testing** — hardcoded 1-token prompt. Must fix to read from argv/stdin before any meaningful comparison can happen.

## Phase 28b Plan
1. Fix F32 waste + memory leak
2. Fix preill N>1 fallback kernel
3. Fix gen_text.c prompt
4. Cos-sim: GPU SSM vs CPU SSM
5. Cos-sim: full inference vs llama.cpp
6. Profile SSM GPU speed
