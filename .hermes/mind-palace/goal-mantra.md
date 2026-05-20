# Goal Mantra — May 20, 2026 (Phase 25 — Fused Quant Matmul + SSM Beta/Alpha ✅)

## THE GOAL
**1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.** ✅
**GPU inference: ✅ working at 256k context, 8.5 tok/s decode.**

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| gen_text_gpu | Full 256k, no hang | ✅ (Bug #13 fixed) |
| GPU Q4_0 KV cache | 1,440MB vs 5,120MB FP16 at 256k | ✅ |
| Q4_0 decode speed | 8.1 tok/s Q4_0 vs 7.6 FP16 | ✅ Fused decode attn |
| Q5_K/Q6_K fused matmul | Incremental dequant+dot, no spill | ✅ Phase 25 |
| SSM beta/alpha fused decode | Manual dot + sigmoid/softplus/gate, 1 vs 6 kernels | ✅ Phase 25 |
| External ref | 35.4 tok/s on RTX 4060 Ti (llama.cpp -ncmoe 30) | 🟡 4-7x gap |
| 256k cos-sim vs llama.cpp | Not measured | ❓ Unverified |
| Bottleneck profiling | Guesses, not nsight | ❓ Needed |

## BUILD
```bash
make gen_text_gpu                # GPU inference (Q4_0 KV default)
GPU_Q4_0_KV=0 ./gen_text_gpu     # FP16 KV cache fallback
```

## NEXT
P0 — Fuse conv1d+SiLU+split+L2 norm into single SSM decode kernel
P1 — MoE router on GPU
P1 — Nsight profiling of decode bottlenecks
P2 — Chunked prefill (3-7x at 256k)
