# Goal Mantra — Phase 28k: GPU MoE Analysis Complete, Moving to P1

**Target:** Accept hybrid path (GPU SSM/GQA + CPU MoE). Build MTP spec decode + vision pipeline.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| GPU MoE v5 (Q8_K kernel) | ✅ COMMITTED | 12ad638 — int8 dot, rintf, sm_120 workarounds |
| GPU MoE vs CPU cos-sim | 🟡 0.9888 per-layer | Fundamental, not a bug — different code paths |
| Hybrid path (GPU SSM/GQA + CPU MoE) | ✅ Works at 5.5 tok/s | FORCE_CPU_MOE=1 env var |
| gen_text_mtp | 🟡 Source exists | NOT COMPILED |
| Vision encoder | 🟡 384 LoC | Untested |
| CUDA sm_120 bugs | ✅ Documented | 3 bugs in skill |
| compare_moe_expert tool | ✅ Built | Per-expert CPU vs GPU comparison |
| DA v13 | ✅ Written | Comprehensive analysis in mind-palace/plans/ |

## P0: Complete — GPU MoE analysis done, hybrid path accepted
1. ✅ Q8_K quantization in GPU kernel (v5)
2. ✅ CUDA sm_120 workarounds (extern float smem, thread-0 reduce)
3. ✅ Per-expert comparison tool
4. ✅ DA v13 root cause analysis
5. ✅ GPU MoE disabled by default (use FORCE_CPU_MOE to re-enable)

## P1: MTP Speculative Decode
1. Build gen_text_mtp (`make gen_text_mtp`)
2. Test with regular model first (MTP=0 fallback)
3. Test with MTP model: /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf
4. Verify acceptance rate (~83% at 2 drafts)

## P1: Vision Pipeline  
5. Build test_vision_real
6. Generate test pixels and verify vision→mmproj→text pipeline

## EVERY FIX: compile → test → document → update DA
