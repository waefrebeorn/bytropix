# Prestige Prompt — May 21, 2026 (Phase 28l: P1 Complete, P2 Up)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**
**GPU SSM decode: ~5.5 tok/s (hybrid) | Vision→text: VERIFIED | MTP: 8.5 tok/s**

## Current State
- Hybrid path working: GPU SSM/GQA + CPU MoE = coherent text at 5.5 tok/s
- MTP spec decode working: 8.5 tok/s, 4% acceptance (quantized head limit)
- Vision pipeline VERIFIED: screenshot→patch embed→27 ViT layers→mmproj→text model→logits
- 2 segfault bugs fixed in wubu_vision.c (n_patches_total cap, scores heap alloc)
- GPU MoE v5 committed but 0.9888 per-layer cos-sim is fundamental code-path diff
- CPU gen_text_cpu builds independently of GPU objects

## DA Debunked Claims (Phase 28b docs — now corrected)
| Old Claim | Reality |
|-----------|---------|
| "F32 waste ~2.2 GB" | ✅ Removed in a032a8f |
| "GPU mem leak ~5.5 GB" | ❌ Never existed — free() was fine |
| "Column-major kernel broken" | ❌ It was CORRECT for GGUF layout |
| "gen_text.c hardcoded" | ❌ Already accepts argv[1] |

## Priority Queue

### P2 — Feature Cream
- [ ] GPU RMSNorm + SiLU + gated norm kernels
- [ ] Chunked prefill (3-7x speedup, Qwen2.5-1M)
- [ ] RoPE extrapolation 4x
- [ ] DSA sparse attention (DeepSeek-V3.2)
- [ ] GPU vision encoder kernels (speed up 63s→real-time)
- [ ] Sigmoid gating + load balancing (DeepSeekMoE)

## Key Math Constants
- D_MODEL = 2048, SSM: 16 K-heads × 128, 32 V-heads × 128
- GQA: 16 Q-heads × 256, 2 KV-heads × 256
- V_HIDDEN = 1152, V_HEAD_DIM = 72, V_N_LAYERS = 27
- V_OUT_HIDDEN = 2048 (matches text space)
- Vision: 3D patch 16×16×2, spatial_merge_size=2
- RoPE: IMRoPE sections [11,11,10,0], θ=10M
