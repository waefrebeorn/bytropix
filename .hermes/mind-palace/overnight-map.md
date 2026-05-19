# Overnight Map — May 19, 2026 PM (Growable KV Cache, Smart GPU Gating)

## Active GPU Components
| Component | When | Status |
|-----------|------|--------|
| GQA QKV+RoPE+Attention | prefill or 2048+ ctx | Batched SGEMM, growable KV cache |
| SSM matmuls (qkv, gate) | prefill N>16 | Q5_K/Q6_K quant kernel |
| SSM recurrence | prefill N>16 | cos-sim=1.0 |
| MoE experts | always | IQ2_XXS kernel |
| Output proj | always | GPU SGEMM (0.1ms vs CPU 10ms) |
| SSM conv+norm+gated norm | never | CPU |
| MoE router | never | CPU |

## Decode Speed
- GPU: 8.5 tok/s cold, ~8 tok/s sustained — thermal-stable
- CPU: 7.3 tok/s cold, ~3 tok/s throttled — severe thermal degradation
- GPU wins on laptops due to distributed thermal load across GPU+CPU

## Key Architecture Decisions
1. **Growable KV cache**: start 4096, double on demand. No VRAM waste
2. **Strided-batched SGEMM**: 2 calls instead of 32 for attention
3. **Smart gating**: GPU GQA only when benefit > overhead (2048+ ctx)
4. **CPU baseline**: pure CPU path always available via `gen_text` binary
5. **N>16 threshold**: GPU SSM only for prefill, not token-by-token decode

## Next Bottlenecks for 256k Context
1. **GPU GQA for decode at long context** — gate at 2048 works, but need to verify chunked attention performance at 256k
2. **FP16 KV cache** — halve VRAM, 10GB→5GB at 256k, fits 8GB card
3. **KV cache attention O(n) at 256k** — need sparse/streaming attention
4. **GPU SSM conv+norm kernel** — for faster prefill

## Blockers
- NONE. All components working. Ready for 256k testing.
