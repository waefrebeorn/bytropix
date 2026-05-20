# Prestige Prompt — May 20 Phase 28e (Post-DA Correction)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**
**GPU SSM decode: ~5.9 tok/s | Vision encoder: 384 LoC ported | Q6_K dequant: FIXED**

## Current State
- Q6_K dequant BUG FIXED (was `32.0` offset, constant ~365 output)
- GPU SSM still anti-correlated to CPU (cos-sim -0.66) ⚠️ STATE MANAGEMENT
- CPU SSM matches llama at cos-sim 0.994 (FORCE_CPU_SSM)
- Vision encoder exists: 27-layer 3D ViT, 384 LoC, mmproj→2048 text space
- 8 local-only commits (not pushed to remote)
- CPU gen_text build broken (GPU symbol issue)

## DA Debunked Claims (Phase 28b docs — now corrected)
| Old Claim | Reality |
|-----------|---------|
| "F32 waste ~2.2 GB" | ✅ Removed in a032a8f |
| "GPU mem leak ~5.5 GB" | ❌ Never existed — free() was fine |
| "Column-major kernel broken" | ❌ It was CORRECT for GGUF layout |
| "gen_text.c hardcoded" | ❌ Already accepts argv[1] |

## Priority Queue

### P0 — Fix GPU SSM divergence
- [ ] Layer-by-layer hidden state trace (GPU vs CPU at each step)
- [ ] Fix recurrence state / conv state management
- [ ] Target: cos-sim > 0.99

### P1 — Infrastructure
- [ ] Fix CPU gen_text build
- [ ] Push 8 commits to remote
- [ ] Re-verify cos-sim at current code state

### P2 — Multi-modal vision integration
- [ ] Build + run test_vision_real
- [ ] Verify vision encoder output quality
- [ ] Wire full vision→text pipeline

### P3 — Feature cream (vault synthesis)
- [ ] Sigmoid gating + load balancing (DeepSeekMoE)
- [ ] Chunked prefill (Qwen2.5-1M)
- [ ] RoPE extrapolation 4x
- [ ] DSA sparse attention (DeepSeek-V3.2)
- [ ] MTP speculative decode (DeepSeek-V3)

## Key Math Constants
- D_MODEL = 2048, SSM: 16 K-heads × 128, 32 V-heads × 128
- GQA: 16 Q-heads × 256, 2 KV-heads × 256
- V_HIDDEN = 1152, V_HEAD_DIM = 72, V_N_LAYERS = 27
- V_OUT_HIDDEN = 2048 (matches text space)
- Vision: 3D patch 16×16×2, spatial_merge_size=2
- RoPE: IMRoPE sections [11,11,10,0], θ=10M
