# State — Phase 29a: IQ1_M GPU Kernel + Q4_K GPU Kernel Verified

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**  
**Reference: llama.cpp (libllama.so via ref_dumper)**  
**CUDA: sm_120 (RTX 5050 Blackwell, 13.1 toolkit)**

## CURRENT STATE
| Component | Result | Status |
|-----------|--------|--------|
| CPU-only text (IQ2_M) | 8.9/17.8 tok/s decode/prefill | ✅ **Optimal path** |
| CPU-only text (IQ1_M) | — | ⚠️ Needs CPU fallback fix, quality degraded at 1.90 BPW |
| GPU vision encoder (ViT + MMProj) | GPU ViT 0.52s, total 15.7s | ✅ **GPU accelerated 4.4x** |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ Working |
| GPU SSM/GQA + CPU MoE hybrid | Coherent text at 5.5 tok/s | ✅ Working |
| **GPU IQ1_M quant matmul** | Single-token + batched kernels | ✅ **Verified exact (3e-7)** |
| **GPU Q4_K quant matmul** | Single-token + batched kernels | ✅ **Added** |
| **CPU IQ1_M fallback** | quantized_matmul_from_q8 | ✅ **Added (dequant+SGEMM)** |

## GPU Kernel Status
| Kernel | Type | Single | Batched | Verified |
|--------|------|--------|---------|----------|
| Q5_K | Full | ✅ | ✅ | ✅ |
| Q6_K | Full | ✅ | ✅ | ✅ |
| Q4_K | Full (no qh) | ✅ | ✅ | 🆕 |
| IQ1_M | Grid-lookup | ✅ | ✅ | ✅ Exact vs CPU |

## COMMITS
- c0254c0 — feat(gpu): IQ1_M + Q4_K quant matmul kernels, CPU IQ1_M fallback
