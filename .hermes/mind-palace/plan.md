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

## 🟡 P1: MTP Speculative Decode + Vision
1. **Build gen_text_mtp** (`make gen_text_mtp`)
   - Test with regular model (falls back to single-token)
   - Test with MTP model: /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf
   - Verify: does acceptance rate match DeepSeek's claimed 83% at 2 drafts?
   
2. **Build vision pipeline**
   - build test_vision_real
   - Generate test image pixels
   - Verify 3D ViT encoder → mmproj → text space

## 🟡 P2: Feature Cream
| Feature | Priority | Status |
|---------|----------|--------|
| GPU RMSNorm + SiLU + gated norm kernels | High | Not started |
| Chunked prefill (3-7x speedup, Qwen2.5-1M) | High | Not started |
| RoPE extrapolation 4x | High | Not started |
| Sparse attention (NSA, DeepSeek V3.2) | High | Not started |
| Sigmoid gating + load balancing (DeepSeekMoE) | High | Not started |
| Hamilton KV cache compression 10x | Mid | Not started |
| Entropix dynamic sampling | Mid | Not started |

## P3-P6: Per plan.md (unchanged)
