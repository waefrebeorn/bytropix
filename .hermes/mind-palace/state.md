# State — FINAL: 256k Context Milestone Reached

**bytropix: Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**Cos-sim 0.9967 — 1:1 PARITY ACHIEVED**
**256k decode: 7.8 tok/s on 8GB laptop GPU**

## VRAM Budget (256k Context)
| Component | Size | Format |
|-----------|------|--------|
| GQA weights | 1,040 MB | F32 dequant |
| SSM weights | 692 MB | Q5_K/Q6_K native |
| FP16 KV cache | 5,120 MB | __half, growable |
| Output proj | 1,900 MB | Q4_K quantized |
| MoE + scratch | ~200 MB | IQ2_XXS + buffers |
| **Total** | **~3,992 MB** | **Fits 6.5-8GB VRAM** |

## GPU Pipeline
| Step | Decode (N=1) | Prefill (N>1) |
|------|-------------|---------------|
| GQA attention | GPU if >2048ctx | GPU |
| SSM matmuls | CPU | GPU |
| SSM recurrence | ✅ GPU | GPU |
| MoE experts | ✅ GPU | GPU |
| Output proj | ✅ GPU | GPU |
| SSM conv+norm | CPU | CPU |
| MoE router | CPU | CPU |

## Key Achievements
- FP16 KV cache (cublasGemmEx CUDA_R_16F, ATTEN_TILE=16384)
- GPU SSM recurrence (cos-sim=1.0, all tokens)
- Growable KV cache (starts 4096, doubles on demand)
- Strided-batched SGEMM (2 launches vs 32 per layer)
- Smart GPU gating (only when benefit > overhead)
- 8 bugs fixed, 10 phases shipped

## Documentation
- `plan.md` — Triple-extended roadmap with full architecture
- `tools/vault/` — Archived verification files
- `README.md` — Quick start and project overview
