# State — May 19, 2026 PM — Phases 15-17 Done, GPU Hybrid Active

## GPU Pipeline
| Part | Where | Benefit |
|------|-------|---------|
| GQA QKV + RoPE + Chunked Attention | GPU | Full speed (when weight format issue resolved) |
| GQA attention matmuls | GPU | ❌ Init fails: model uses fused `attn_qkv.weight`, code expects separate Q/K/V |
| SSM quantized matmuls (qkv, gate) | GPU | 3 big matmuls offloaded |
| SSM recurrence (selective scan) | GPU | 🆕 Recurrence GPU kernel (cos-sim=1.0 verified) |
| MoE routed experts (IQ2_XXS gate/up/down) | GPU | 8 experts × 3 matmuls offloaded |
| Output projection (Q4_K 2048×248320) | GPU | Full acceleration |
| SSM conv + norm + gated norm | CPU | Next bottleneck |
| MoE router + shared expert | CPU | Fast enough |
| GQA scores, attention softmax | CPU | Fast enough |

## Speed
- **CPU decode: 4.4 tok/s** (gen_text_gpu GPU=0)
- **GPU decode: 6.8 tok/s** (GPU for SSM recurrence + matmuls + MoE + output proj)
- **Speedup: +55%** from GPU SSM recurrence kernel (was 3.3 tok/s before)

## Key GPU Recurrence Details
- 32 V-heads, 128 threads per block, state matrix [128][128] in global memory (64KB/head)
- Shared memory: 2.5KB/block (too large for shared: 64KB state)
- cos-sim 1.0 vs CPU, max err 1e-6 (FP rounding only)
- GQA GPU still broken: model uses fused `attn_qkv.weight`, code expects separate Q/K/V
## Committed
3 commits pushed: `feat(moe): wire GPU MoE into forward pass`, `perf(moe): pre-alloc buffers`, MoE kernel v3 with pre-alloc buffers.
