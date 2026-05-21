# Goal Mantra — Phase 28l: P1 Complete, P2 Up

**Target:** Hybrid path (GPU SSM/GQA + CPU MoE) working. MTP + Vision verified. Next: feature cream.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| GPU SSM/GQA + CPU MoE | ✅ 5.5 tok/s | Coherent text |
| MTP spec decode | ✅ 8.5 tok/s | 4% acceptance (quantized head) |
| Vision→text pipeline | ✅ Verified | 256×256→128 patches→logits, no NaN |
| Vision encoder | ✅ Verified | 63.7s CPU, 2 segfault bugs fixed |
| GPU MoE v5 | ✅ COMMITTED | 12ad638, fundamental 0.9888 cos-sim |
| DA v13 | ✅ Written | Comprehensive analysis |

## P0: Complete — GPU MoE analysis done, hybrid path accepted
1. ✅ Q8_K quantization in GPU kernel (v5)
2. ✅ CUDA sm_120 workarounds (extern float smem, thread-0 reduce)
3. ✅ Per-expert comparison tool
4. ✅ DA v13 root cause analysis
5. ✅ GPU MoE disabled by default (use FORCE_CPU_MOE to re-enable)

## P1: Complete
1. ✅ MTP spec decode — gen_text_mtp working at 8.5 tok/s
2. ✅ Vision pipeline — screenshot→encoder→mmproj→text→logits verified

## P2: Feature Cream (up next)

## EVERY FIX: compile → test → document → update DA
