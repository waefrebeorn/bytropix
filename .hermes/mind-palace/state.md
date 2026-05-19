# State — May 19, 2026 PM — Phase 16-17 Done, Growable KV Cache, Smart GPU Gating

## GPU Pipeline
| Part | Where | Benefit |
|------|-------|---------|
| GQA QKV + RoPE + Attention | GPU 🤖 prefill+N>2048ctx | Batched SGEMM, growable cache |
| SSM quantized matmuls (qkv, gate) | GPU 🤖 N>16 | Quantized kernel |
| SSM recurrence (selective scan) | GPU 🤖 N>16 | cos-sim=1.0 verified |
| MoE routed experts (IQ2_XXS) | GPU ✅ Always | IQ2_XXS kernel |
| Output projection (Q4_K 2048×248320) | GPU ✅ Always | ~0.1ms vs CPU ~10ms |
| SSM conv + norm + gated norm | CPU | Next target |
| MoE router + shared expert | CPU | Fast enough |
| GQA scores + softmax | CPU | Fast enough |

## Speed (first cold run, no thermal degrade)
- **CPU decode (gen_text): 7.3 tok/s** — thermal throttles to ~3 tok/s after 1-2 runs
- **GPU decode (gen_text_gpu): 8.5 tok/s** — more stable, ~8 tok/s sustained
- GPU advantage: output proj (0.1ms vs 10ms CPU) + distributed thermal load

## Growable KV Cache
- Starts at 4096, doubles on demand up to max_ctx
- VRAM: 160MB initial + 10 layers = 1.6GB → grows to 10GB at 256k
- Smart gate: GPU GQA only when cache_len>2048 or N>1 (prefill)

## Strided-Batched Attention
- Chunked attention direct path: 2 strided-batched SGEMMs instead of 32 per layer
- 20 kernel launches per decode (was 320)

## GPU SSM Recurrence Kernel
- 32 V-heads × 128 threads, state [128][128] in global memory (64KB/head)
- cos-sim 1.0 verified vs CPU, max err 1e-6
- Active only for N>16 (prefill)
- Wired via ssm_layer_weights.gpu_ssm_state + goto gpu_rec_done

## Key Learnings
- Single-token GPU offload has negative ROI (transfer/sync > compute savings)
- GPU GQA only worthwhile at 2048+ context or N>1 prefill
- Thermal throttling on laptop CPU (3x slowdown) is a bigger issue than GPU bottleneck
- KV cache at 256k F32 = 10GB → must use FP16 or growable approach

## Committed
5 commits: `feat(gpu): growable KV cache, strided-batched attention, smart GPU gating`
