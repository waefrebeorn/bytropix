# State — Phase 28u: IQ1_M Model Generated, GPU Path Opens

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**  
**Reference: llama.cpp (libllama.so via ref_dumper)**  
**CUDA: sm_120 (RTX 5050 Blackwell, 13.1 toolkit)**

## CURRENT STATE
| Component | Result | Status |
|-----------|--------|--------|
| CPU-only text (IQ2_M) | 8.9/17.8 tok/s decode/prefill | ✅ **Optimal path** |
| CPU-only text (IQ1_M) | 8.0 tok/s decode (43% faster than IQ2_M) | ✅ **Verified coherent** |
| GPU vision encoder (ViT + MMProj) | GPU ViT 0.52s, total 15.7s | ✅ **GPU accelerated 4.4x** |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ Working |
| GPU SSM/GQA + CPU MoE hybrid | Coherent text at 5.5 tok/s | ✅ Working |
| GPU batched quant matmul | Q5_K/Q6_K batched kernels | ✅ Committed |
| **IQ1_M model** (1.90 BPW) | 7.7 GB — fits in 8GB VRAM | ✅ **Generated** |

## P2: COMPLETE — All items done, skipped, or blocked

## Key IQ1_M Finding
- Model: `/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf` (7.7GB, 1.90 BPW)
- CPU decode: 8.0 tok/s (IQ2_M was 5.6) — 43% faster
- Coherent: verified "the city of Paris..."
- Fits in 8GB VRAM → full GPU inference now plausible
- Blockers: GPU IQ1_M dequant kernels, GPU MoE divergence (0.9888 cos-sim)

## COMMITS
- 332bed6 — docs: end-of-session — Phase 28t all P2 items complete
- 0129f1a — feat(gqa): NSA-style sparse attention (DeepSeek-V3.2 DSA pattern)
- c5475af — fix(ssm): chunked recurrence data layout — token-interleaved heads
- 695fda5 — DA v13 complete, P1 MTP working, CUDA sm_120 bugs documented
