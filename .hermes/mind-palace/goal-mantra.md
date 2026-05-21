# Goal Mantra — Phase 28o: P2 Hardware Utilization

**Target:** CPU-optimal path (8.9 tok/s decode) stable. GPU vision pipeline accelerated 4.4x (15.7s total). P2 feature cream: hardware utilization, RoPE extrapolation, chunked prefill, sparse attention, sigmoid gating.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| GPU vision pipeline | ✅ 15.7s total | GPU ViT 0.52s + GPU MMProj cuBLAS + CPU text 6.3s |
| GPU hybrid text | ⚠️ NET-NEGATIVE | CPU-only 2-5x faster. GPU MoE 0.9888 cos-sim is FUNDAMENTAL |
| MTP spec decode | ✅ 8.5 tok/s | 4% acceptance (quantized head — IQ2_M MTP model) |
| CPU-only text | ✅ 8.9/17.8 tok/s | Optimal path for all text inference. Stable, coherent |
| Vision full pipeline | ✅ Verified | No NaN/Inf, logit range [-10.8, 14.1] |

## P0-P1: Complete
1. ✅ GPU MoE root cause (DA v13): 0.9888 cos-sim is code-path diff, NOT fixable
2. ✅ Hybrid path accepted (GPU SSM/GQA + CPU MoE = coherent at 5.5 tok/s)
3. ✅ GPU MoE disabled by default (FORCE_CPU_MOE)
4. ✅ MTP spec decode — gen_text_mtp working at 8.5 tok/s
5. ✅ Vision pipeline — screenshot→encoder→mmproj→text→logits verified
6. ✅ 2 segfaults fixed in wubu_vision.c
7. ✅ GPU GQA batched prefill (C=N)
8. ✅ Batched quant matmul (Q5_K/Q6_K)

## P2: Feature Cream — Chunked SSM Bug Fixed
1. ✅ CUDA sm_120 bug documentation → skill (in DA v13)
2. ✅ Chunked prefill data layout bug FIXED (c5475af). CS=1 exact match.
3. Llama.cpp inline hooks for reference data (replace llama-cli)
4. GPU RMSNorm + SiLU kernels (exist, not wired)
5. NSA sparse attention (DeepSeek-V3.2, high effort)
6. Sigmoid gating + load balancing (DeepSeekMoE)
7. FP8 Tensor Cores (sm_120 native, 2x throughput)

## EVERY FIX: compile → test → document → update DA
