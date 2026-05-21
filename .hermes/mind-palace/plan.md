# Plan — Phase 28k+: GPU MoE Analysis Complete, P1 Active

## 🔴 P0: GPU MoE Hidden State Divergence — COMPLETE
Root cause identified (DA v13): 0.9888 per-layer cos-sim is FUNDAMENTAL — not a single bug. Different code paths produce different rounding. Hybrid path accepted.

### What Was Done
1. ✅ v5 Q8_K kernel: quantize x to Q8_K, use int8 dot product
2. ✅ CUDA sm_120 bugs: 3 workarounds applied
3. ✅ Per-expert compare tool: compare_moe_expert
4. ✅ DA v13: comprehensive root cause analysis
5. ✅ GPU MoE disabled by default (FORCE_CPU_MOE)

### What Was Learned
- 0.32% running cos-sim error → flips token selection in 240K vocab
- Hybrid path (GPU SSM/GQA + CPU MoE) produces coherent text at 5.5 tok/s
- Q8_K quantization is correct but doesn't fix the ~1.1% per-layer error
- For 1:1 parity, would need CPU quantized_matmul ported to GPU (3-5 sessions)

## 🟡 P1: MTP Speculative Decode + Vision — COMPLETE
1. ✅ **Build gen_text_mtp** — working at 8.5 tok/s, 4% acceptance (quantized head)
2. ✅ **Vision pipeline** — screenshot→encoder→mmproj→text→logits verified
   - 2 segfault bugs fixed in wubu_vision.c
   - 256×256 → 128 patches × 2048, no NaN, logit range [-10.8, 14.1]
   - test_vision_real builds with GPU_SUPPORT

## 🟡 P2: Feature Cream
| Feature | Priority | Status |
|---------|----------|--------|
| GPU GQA batched prefill fix | High | ✅ Done (C=N, sub-batch fallback) |
| **GPU vision encoder kernels** | **High** | **✅ GPU ViT 0.52s (122x faster), NaN=0** |
| SSM forward_full C>1 path (batched SSM prefill) | High | ❌ Accuracy divergence, needs DA |
| GPU RMSNorm + SiLU + gated norm kernels | High | 🔲 Kernels exist, not wired |
| Chunked prefill (3-7x speedup, Qwen2.5-1M) | High | 🔲 Infrastructure exists |
| RoPE extrapolation 4x | High | Not started |
| Sparse attention (NSA, DeepSeek V3.2) | High | Not started |
| Sigmoid gating + load balancing (DeepSeekMoE) | High | Not started |
| Hamilton KV cache compression 10x | Mid | Not started |
| Entropix dynamic sampling | Mid | Not started |

## P3-P6: Per plan.md (unchanged)
