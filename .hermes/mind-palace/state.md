# State — Phase 21: Sliding Window Attention for 256k GQA

**bytropix: Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**Decode: 9.3 tok/s — Prefill: 18.6 tok/s — Sliding window GQA: 16→1 tile at 256k**
**GQA_WINDOW env var: 0=full attention (default), N=attend to last N tokens only**

## VRAM Budget (256k Context)
| Component | Size | Format |
|-----------|------|--------|
| GQA weights (F32) | 1,040 MB | cuBLAS SGEMM |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K native on GPU |
| SSM F32 weights (small) | ~30 MB | beta/alpha/dt_bias/a/conv1d/norm |
| SSM GPU conv_state | 2.3 MB | [3, 8192] × 30 layers persistent |
| FP16 KV cache | 5,120 MB | __half, growable |
| Output proj (Q4_K) | 1,900 MB | quantized GPU kernel |
| SSM scratch | 49 MB | reusable intermediates |
| MoE + scratch | ~460 MB | cache(259MB) + scratch(200MB) |
| **Total** | **~8,293 MB** | **~Fits 8GB VRAM** |

## Key Achievements
- **Phase 21**: Sliding window attention — `GQA_WINDOW=N` skips tiles outside last N tokens. At 256k with window=16384: 1 tile vs 16 tiles per GQA layer = ~16x fewer SGEMM calls for attention.
- **Phase 20**: MoE expert cache on GPU — 259MB cache, eliminates H2D on routing stability
- **Phase 19**: Batched prefill via `wubu_cuda_ssm_parallel_scan` — 18.6 tok/s (+59%)
- **Phase 18**: Full GPU SSM pipeline — only 2 transfers/layer
- 10+ phases shipped, 8+ bugs fixed

## GPU Pipeline Status
| Component | GPU | Notes |
|-----------|-----|-------|
| SSM forward (all 15 steps) | ✅ | 2 transfers/layer, ~0.9ms |
| GQA attention (FP16 KV) | ✅ | Sliding window via GQA_WINDOW env var |
| MoE expert matmuls | ✅ | Cached, zero-H2D on stable routing |
| Output projection (Q4_K) | ✅ | 2048×248320, ~0.1ms |
| MoE router (F32) | ⬜ CPU | Small (2048×256), ~10μs |

## Remaining Targets
1. **Unified SSM forward kernel** — fuse all SSM steps into single kernel
2. **Sparse attention with global tokens** — add global position attention to sliding window

