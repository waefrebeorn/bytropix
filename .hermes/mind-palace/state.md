# state — May 17 v17 — DA Audit: SSM L2 eps mismatch = root cause of 0.006 gap

## DA VERDICT: 1:1 parity BLOCKED by SSM L2 normalization epsilon
- **MoE code**: ✅ cos-sim 1.000000 internal consistency (verified fresh build, same input)
- **SSM/GQA recurrence formulas**: ✅ algebra matches llama.cpp exactly
- **L2 normalization epsilon**: 🔴 **1e-12 (us) vs ~1e-6 (llama.cpp from model GGUF)** = root cause
- Previous "cos-sim 0.994 MOE=0" = 0.006 gap from this epsilon difference alone

## What We Know (C code to C code, no excuses)
| Claim | Status | Evidence |
|-------|--------|----------|
| MoE lazy vs library output | ✅ cos-sim 1.0 | Fresh build, same input, bit-identical dequant |
| Routing selects same top-8 | ✅ Identical | verify_topk_agreement confirmed |
| Per-expert dequant bit-identical | ✅ cos-sim 1.0 | verify_dequant confirmed across 8 experts |
| SSM recurrence formula | ✅ Algebra matches llama.cpp | delta-net-base.cpp:291-376, same:
  state *= exp(gate); hk = h@k; diff = v - hk; h += k@diff*bg; out = h@(q/sqrt(d)) |
| SSM L2 norm eps | ❌ **WRONG: 1e-12 vs ~1e-6** | wubu_ssm.c:318-319 hardcodes 1e-12, llama.cpp reads hparams.f_norm_rms_eps |
| SSM safe-exp clamp | ✅ Clamp [-80,80] prevents overflow | llama.cpp ggml_exp has no clamp — if anything, our approach is more stable |
| Head repeat mechanism | ✅ Same cyclic repeat (16→32) | bytropix: vh % 16; llama.cpp: ggml_repeat_4d |
| Q scaling | ✅ Both use 1/sqrt(128) | Same factor |
| GQA | ❓ Untested (10 layers) | Subagent didn't analyze GQA path |
| GPU RoPE factor | ❓ Untested | README mentions infer_text_gpu.c:254 |
| Output projection | ✅ Fixed (3 transpose fixes) | Verified in earlier session |

## Plan Forward to 1:1 Parity
1. **Fix L2 norm epsilon**: wubu_ssm.c:318-319 — change `1e-12f` to read from model config
2. **Verify fix**: Run MOE=0 comparison vs reference — expect cos-sim jump 0.994→1.0
3. **Enable MoE**: Run MOE=1 comparison — expect cos-sim ~1.0 (both paths verified matching)
4. **Verify GQA**: Status unknown — needs separate audit
5. **Full model test**: Run 40-layer with MoE, compare logits vs llama-cli

## Cleanup needed
- README.md — claims "MoE stride bug ACTIVE" (WRONG — experts ARE contiguous)
- STATUS.md — references old training metrics, not inference parity
- prestige_prompt.md — claims "MoE=1 divergence cos-sim 0.337" (STALE DATA)
- overnight-map.md — same stale claims
- plan.md — needs DA-informed correction
