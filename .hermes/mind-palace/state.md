# state — May 17 v13 — MoE: all subcomponents verified correct, root cause unknown

## Status
- **SSM/GQA verified CORRECT** ✅
  - MOE=0: cos-sim 0.998 vs reference MOE=0
  - RMSNorm, RoPE, weight dequant, access patterns all verified
- **MoE output: still wrong** ❌
  - Layer 0 MoE contribution: rms≈0.027 (30x too small vs pass-through normed)
  - 40-layer final: cos-sim 0.107 vs reference MOE=1
  - Reference MOE=0 vs MOE=1: cos-sim 0.007 — MoE completely determines output direction

## Verified Correct (all match ggml reference)
- **IQ2_XXS dequant** ✅ matches ggml for expert 0 AND expert 64 (100K/100K)
- **IQ3_XXS dequant** ✅ matches ggml for expert 0 AND expert 64 (100K/100K, 0 diffs)
- **Raw sizes** ✅ match ggml_row_size exactly
- **Expert extraction** ✅ experts 0 and 64 have different weights (correct offset calc)
- **Expert forward** ✅ gate rms=0.4255, up rms=0.3550, act rms=0.1045, out rms=0.0262
- **Router** ✅ selects reasonable experts (e=64 wgt=0.585, etc.)
- **Shared expert** ✅ Q5_K/Q6_K weights, correct computation
- **Weight access patterns** ✅ k + j*D_MODEL correct for GGML dims
- **All model metaparameters** ✅ expert_count=256, expert_used_count=8, D_FF=512, D_MODEL=2048

## Findings
- Reference MOE=0 vs MOE=1 have cos-sim 0.007 — MoE entirely determines output
- The IQ2_XXS weights have rms≈0.005 (2-bit precision, inherently small values)
- This leads to MoE output rms≈0.027 per layer vs pass-through normed rms≈0.89
- The small MoE correction per layer gets amplified through recurrence
- Reference uses same model file (no fused gate_up_exps tensor, separate gate/up)
- No shared expert gate_inp_shexp in model (our sh_gate_proj is NULL)

## Remaining Hypotheses
1. Full 1M-element dequant test needed (only tested 100K)
2. IQ4_XS dequant for last 3 layers (type 19) not verified
3. **OUTPUT GENERATION TEST** — model generates "Hello painting" from "Hello" with MOE=1
   - Need to compare against reference generation

## Debug Patches Still Active
- infer_text.c: has expert weight raw dump (dequant_multi_expert_contiguous)
- infer_text.c: has dequantized gate weight dump (layer 0, expert 0, type 16)
- infer_text.c: has lazy_moe_decode debug print (gate/up/act/out rms per expert)
- llama.cpp: qwen35moe.cpp has post-MoE output marked as output tensor

## Files
- `/tmp/debug_router.bin` — topk indices for layer 0: [64, 161, 112, ...]
- `/tmp/ggml_exp64_gate_deq.bin` — ggml dequant, expert 64 gate, rms=0.00979
- `/tmp/ggml_exp0_down_deq.bin` — ggml dequant, expert 0 down (IQ3_XXS)
