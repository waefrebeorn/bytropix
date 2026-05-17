# bytropix Plan — May 17 v16 (All components verified, MoE divergence = recurrent amp)

## CURRENT STATUS
- **SSM/GQA path**: ✅ cos-sim 0.994 logits vs reference (MOE=0)
- **All dequant types**: ✅ verified exact vs ggml (IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K)
- **MoE=1 logits**: ❌ cos-sim 0.337 vs reference (recurrent amplification hypothesis)
- Model generates text with MoE enabled but trajectory diverges from reference

## Phase 0.5: First-Token Parity

### DONE
- [x] RoPE MRoPE section dimension fix (22/22/20, was 64)
- [x] Output projection transpose fix (3 places)
- [x] Reference extraction tool (run_ref_moe0)
- [x] RMSNorm verified cos-sim 1.0 vs numpy
- [x] All dequant types verified exact vs ggml
- [x] SSM/GQA path verified at cos-sim 0.994 vs reference
- [x] MoE path isolated as sole source of divergence
- [x] Top-k renormalization confirmed matches reference
- [x] Reference runs 100% CPU (no GPU divergence)

### CURRENT: Trace divergence onset
- [ ] Compare MOE=1 vs MOE=0 per-layer residuals to see where MoE correction first diverges
- [ ] If layer-0 divergence: compare `moe_expert_forward_lazy` vs `build_moe_ffn` order of ops
- [ ] If layer 2-3+ divergence: likely recurrent amplification (fundamental to 40-layer SSM)

### IF FIXABLE
- [ ] Match llama.cpp MoE graph computation order exactly
- [ ] Softmax/top-k numerical precision matching (float vs double?)

### DOWNSTREAM Optimizations
- [ ] OpenMP on weight loading paths
- [ ] VRAM cleanup in SIGINT handler
- [ ] GPU RoPE 0.25x factor in infer_text_gpu.c:254

## Phase 1: Multi-token prefill parity
- Blocked on first-token correctness.
