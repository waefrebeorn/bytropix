# bytropix Plan — May 17 v17 (MoE cos-sim 1.0, stale data corrected)

## STATUS
- **SSM/GQA path**: ✅ cos-sim 0.994 logits vs reference (MOE=0, needs fresh verification)
- **MoE path**: ✅ cos-sim 1.000000 internal consistency (no bug found)
- All previous "divergence" numbers were from stale binary comparisons
- Model generates plausible output with MoE enabled

## DONE
- [x] RoPE MRoPE section dimension fix (22/22/20, was 64)
- [x] Output projection transpose fix (3 places)
- [x] Reference extraction tool (run_ref_moe0)
- [x] RMSNorm verified cos-sim 1.0 vs numpy
- [x] All dequant types verified exact vs ggml
- [x] SSM/GQA path verified at cos-sim 0.994 vs reference
- [x] MoE lazy vs library: cos-sim 1.000000 verified
- [x] Top-k agreement: identical expert selection
- [x] Dequant bit-identity: verified across 8 experts
- [x] Stale binary issue identified — rebuild infer_text before any comparison

## Cleanup
- [/] Remove stale debug files from /tmp/
- [/] Commit current state with corrected findings
- [ ] Decide next direction: fresh SSM/GQA vs reference comparison with rebuilt infer_text?
