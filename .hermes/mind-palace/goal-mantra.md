# Goal Mantra — Phase 28r: P2.3 Chunked SSM Fixed + Wired

**Target:** CPU-optimal path (8.9 tok/s decode) stable. P2.3 chunked SSM data bug fixed, wired into inference. CS=1 exact. CS>1 FP-limited.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| GPU vision pipeline | ✅ 15.7s total | GPU ViT 0.52s + GPU MMProj cuBLAS + CPU text 6.3s |
| GPU hybrid text | ⚠️ NET-NEGATIVE | CPU-only 2-5x faster |
| MTP spec decode | ✅ 8.5 tok/s | 4% acceptance |
| CPU-only text | ✅ 8.9/17.8 tok/s | Optimal path. Stable, coherent |
| Chunked SSM CS=1 | ✅ Exact | 4e-8 output diff, 3e-7 state diff |
| Chunked SSM CS>1 | ⚠️ FP-limited | Accumulates across 30 SSM layers → wrong tokens |

## P0-P2: Complete
1. ✅ GPU MoE root cause (DA v13) — fundamental code-path diff
2. ✅ MTP spec decode — gen_text_mtp at 8.5 tok/s
3. ✅ Vision pipeline — screenshot→encoder→mmproj→text→logits
4. ✅ GPU GQA batched prefill (C=N)
5. ✅ Batched quant matmul (Q5_K/Q6_K)
6. ✅ RoPE extrapolation 4x — `ROPE_SCALE_FACTOR=0.25`
7. ✅ Chunked SSM data layout bug FIXED, wired into forward pass

## P2 Remaining
1. Llama.cpp inline hooks for reference data
2. NSA sparse attention (DeepSeek-V3.2, high effort)
3. FP8 Tensor Cores (sm_120 native, needs GPU data-movement solution)

## BUILD
`make gen_text_cpu` for CPU-only inference binary
`FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N` — sequential path
