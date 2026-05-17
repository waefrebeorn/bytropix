# bytropix Plan — May 17 v10 (MoE expert layout bug = root cause)

## CURRENT STATUS
- **Output proj transpose FIXED** ✅
- **MoE expert layout bug FOUND** via DA audit ✅
- **Full model still wrong** — expected, MoE weights were garbage
- GQA, SSM, RMSNorm, RoPE formulas have NOT been ruled out yet

## Phase 0.5: First-Token Parity

### DONE
- [x] Output projection transpose fix (3 places)
- [x] Reference extraction tool (dump_llama_logits + hidden)
- [x] DA audit of ALL claimed verifications (found 3 false ✅s)
- [x] RMSNorm verified cos-sim 1.0 vs numpy

### PRIORITY 1: Fix MoE Expert Layout
- [ ] Fix `dequant_one_expert_contiguous` → stride-extract per expert
- [ ] Fix `dequant_multi_expert_contiguous` similarly
- [ ] Fix shared expert dequant if also interleaved (check dims)
- [ ] Rebuild and compare full model output vs reference

### IF STILL WRONG After MoE Fix
- [ ] Dump layer 0 residual from our model vs llama.cpp
- [ ] Compare element-by-element, find first divergence
- [ ] If divergence at layer 0: SSM forward wrong (wubu_ssm_forward)
- [ ] If divergence at layer 1+: compounding error from MoE fix

### DOWNSTREAM Optimizations
- [ ] OpenMP on weight loading paths
- [ ] VRAM cleanup in SIGINT handler (llama.cpp reference)
- [ ] Fix GPU RoPE 0.25x factor in infer_text_gpu.c:254

## Phase 1: Multi-token prefill parity
- Blocked on first-token correctness.

## Phase 2-5
- Blocked on Phase 1.
