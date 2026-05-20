# State — Phase 28f: GPU Output Proj + State Sync FIXED, Q5_K Quant Matmul Bug Found

**bytropix: text + vision multi-modal inference engine**
**Phase 28f: GPU output projection ✅, SSM state sync ✅, hybrid inference ~coherent**

## Achieved This Session

1. **GPU output projection FIXED** — Q4_K dequant stored as [D][V] row-major with wrong stride (ceil(V/256) instead of ceil(D/256)). Now stores as [V][D] row-major with CUBLAS_OP_T ld=D. All 248k logits non-zero. ✅
2. **SSM state sync FIXED** — Hybrid prefill updates CPU ssm_state/conv_state. Decode via forward_full or hybrid recurrence used GPU state initialized to zero. Added CPU→GPU sync after hybrid prefill and before decode. Added GPU→CPU after forward_full decode. ✅
3. **GPU recurrence kernel VERIFIED** — Isolated test shows ssm_recurrence_kernel produces cos-sim 1.000 with CPU (same input). Not the source of divergence. ✅
4. **GPU L2 norm VERIFIED** — cos-sim 1.000 with CPU. ✅
5. **GPU head repeat VERIFIED** — cos-sim 1.000 with CPU. ✅

## 🔴 ROOT CAUSE: GPU Q5_K Quant Matmul Bug
**GPU wubu_cuda_quant_matmul for Q5_K produces wrong results (cos-sim ~0.16 vs CPU Q8_K path).**

Evidence:
- x=[1,0,0,...] input: GPU returns -0.028 for column 0, CPU dequant gives -0.004 (first weight element)
- Full random x: qkv cos-sim 0.164 between GPU and CPU
- The dequant logic appears identical between GPU kernel and host function
- My host-side Q5_K dequant of the same block bytes produces correct -0.004 value
- GPU kernel reading same data returns -0.028 for the same computation

Suspect: fp16_to_fp32_dev converts F16 denormals to zero (line 19-20 in gpu_quant_matmul.cu), but the d/dmin values in this model are normal F16, so shouldn't be the issue. Deeper investigation needed.

## Hybrid Inference Status (Current Workaround)
- Prefill (5 tokens): hybrid path (GPU qkv/gate via buggy Q5_K matmul + CPU conv/recurrence/norm)
- Decode: hybrid recurrence (CPU conv/norm, GPU recurrence kernel via wubu_gpu_set_ssm_hybrid)
- Output: ~coherent English (generates "Par" for Paris-style prompts)
- Not token-identical to CPU due to Q5_K quant matmul divergence

## Next Steps
- Fix Q5_K gpu_quant_matmul.cu quant_matmul_q5_k_kernel
- After fix: re-enable forward_full for decode
- Verify cos-sim > 0.99 vs CPU

## Quick Build
```bash
make gen_text_gpu
GPU_BATCH=5 GPU=1 MAX_CTX=4096 ./gen_text_gpu "prompt" 20 40
```
