# State — Phase 19: Batched Prefill via Parallel Scan

**bytropix: Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**GPU SSM: Only 2 transfers per layer (H2D input + D2H output)**
**Prefill: 18.6 tok/s — Decode: 8.7 tok/s — 256k capable, 8GB VRAM**

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
| MoE + scratch | ~200 MB | IQ2_XXS + buffers |
| **Total** | **~8,033 MB** | **~Fits 8GB VRAM** |

## Key Achievements
- **Phase 19**: Batched prefill via `wubu_cuda_ssm_parallel_scan` — C>1 tokens processed in single GPU call. Prefill: 11.7→18.6 tok/s (+59%).
- **Phase 18c**: GPU conv_state + GPU K-head repeat. Persistent GPU conv_state. repeat_kheads_kernel eliminates 10 H2D/D2H transfers per SSM layer.
- **Phase 18b**: FP16 chunked attention GPU softmax. online_softmax_row_kernel replaces host-side loop. normalize_attn_kernel, batched softmax_kernel, pre-allocated MoE d_x.
- **Phase 18**: Full GPU SSM pipeline — all 15 steps on GPU.
- FP16 KV cache, growable KV cache, strided-batched SGEMM (2 launches vs 32), smart GPU gating.
- 8 bugs fixed, 10+ phases shipped.

## GPU SSM Transfer Profile
| Before | After | Savings |
|--------|-------|---------|
| 7 H2D/D2H transfers/layer | 2 transfers/layer | **-116KB/layer, -3.5MB/decode** |

## Remaining Targets
1. **MoE expert cache on GPU** — cache last-8 active experts per layer, skip H2D when routing stable
2. **Sparse/streaming attention** — O(n·k) for >256k GQA attention
3. **MoE router on GPU** — removes CPU from forward
4. **Unified SSM forward kernel** — fuse all SSM steps into single kernel
