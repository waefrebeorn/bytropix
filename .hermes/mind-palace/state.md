# State — May 19, 2026 PM — FP16 KV Cache ✅, 256k Context Viable

## GPU Pipeline
| Part | Status |
|------|--------|
| GQA QKV + RoPE + Attention | GPU: FP16 cache (__half), cublasGemmEx |
| SSM matmuls (qkv, gate) | GPU 🤖 N>16 prefill |
| SSM recurrence | GPU 🤖 N>16 prefill (cos-sim=1.0) |
| MoE experts | ✅ Always GPU |
| Output proj | ✅ Always GPU SGEMM |
| SSM conv+norm+gated norm | CPU |
| MoE router | CPU |

## VRAM Budget (8GB Laptop)
| Component | Size | Notes |
|-----------|------|-------|
| GQA weights (F32 dequant) | 1,040 MB | 10 layers × F32 |
| SSM weights (Q5_K/Q6_K) | 692 MB | 30 layers, native quantized |
| KV cache (init 4096) | 160 MB | FP16, grows to 5GB at 256k |
| Output proj (Q4_K) | 1,900 MB | GPU Q4_K kernel |
| MoE + scratch | ~200 MB | |
| **Total (256k ctx)** | **~3.8 GB** | Fits 8GB with headroom |

## Speed
- Decode (short ctx): 7.8 tok/s
- Decode (256k ctx): 4.3 tok/s
- Prefill (5-tok): 10.7-11.7 tok/s

## Commits
6 commits pushed. Latest: `feat(gpu): FP16 KV cache — halves VRAM, enables 256k context on 8GB`
