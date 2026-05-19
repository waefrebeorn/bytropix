# Overnight Map — May 19, 2026 PM (Phases 15-17 Active, GPU Hybrid)

## Active GPU Components
| Phase | Component | GPU? | Status |
|-------|-----------|------|--------|
| 15 | GQA forward (QKV, RoPE, attention) | ✅ Full GPU | shipped |
| 16 | SSM matmuls (qkv, gate) | ✅ GPU matmul | shipped |
| 17 | MoE routed expert compute | ✅ GPU IQ2_XXS | shipped |
| 13 | Output projection (Q4_K) | ✅ GPU | shipped |
| — | SSM conv, norm, recurrence | ❌ CPU | bottleneck |
| — | MoE router + shared expert | ❌ CPU | bottleneck |

## Decode Speed
- CPU (gen_text): 6.4 tok/s
- GPU (gen_text_gpu GPU=1): 3.2 tok/s
- **GPU speed is 2x slower than CPU** — SSM conv/norm/recurrence still on CPU dominates

## Optimizations This Session
- Pre-allocated MoE GPU buffers (eliminated 320 cudaMalloc/Free per decode)
- Single per-expert kernel launch (avoids shared memory occupancy issue of batching)
- Original per-expert approach restored with pre-alloc buffers

## Next Direction Options
1. **Port SSM conv + norm + recurrence to GPU** — biggest potential speedup (target ~6 tok/s)
2. **Port SSM output projection to GPU** — weights already on GPU
3. **256k context GQA** — sparse/streaming attention for long context
