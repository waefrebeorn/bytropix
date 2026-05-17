# state — May 17 v10 — MoE expert layout IS THE BUG (interleaved, not contiguous)

## Status
- **Output projection TRANSPOSE** — FIXED in 3 places ✅
- **MoE expert tensor layout WRONG** — ROOT CAUSE found via DA audit ❌
  - `blk.0.ffn_gate_exps.weight` dims = [2048, 512, 256]
  - Expert index (256) = INNERMOST dim, data is interleaved per (i,j) position
  - `dequant_one_expert_contiguous` reads `eid * raw_per_exp` — treats as contiguous but data is stride-interleaved by expert
  - Each IQ2_XXS block (66 bytes, 256 values) = ALL experts at ONE (i,j) position
  - Fix: dequant each block, extract float[eid], store at position b
  - This affects ALL MoE layers: ffn_gate_exps, ffn_up_exps, ffn_down_exps
- **Full model output still WRONG** (cos-sim -0.001 after output proj fix) — expected, MoE is still processing wrong weights
- **Reference extraction tool works** ✅

## DA Audit: All Components Re-Verified

### What was ACTUALLY verified this session
1. RMSNorm formula: cos-sim 1.0 vs numpy ✅
2. llama_get_embeddings_ith: works for hidden state extraction ✅
3. RoPE CPU formula: correct (MRoPE [11,11,10,0], no 0.25x) ✅
4. Output projection TRANSPOSE: FOUND and FIXED ✅

### What was WRONGLY claimed as verified (DA findings)
1. **"MoE expert dequant correct"** — ❌ EXPERT DATA IS INTERLEAVED, contiguous extraction produces garbage
2. **"Output projection verified"** — ❌ Was TRANSPOSED, only caught now
3. **"SSM formulas verified correct"** — ❓ Only "reasonable values" in debug dumps, never compared element-by-element vs llama.cpp
4. **"GQA algorithm verified correct"** — ⚠️ Partly true: `wubu_gqa_forward` verified vs numpy, but infer_text.c's INLINE GQA was never separately verified (though code is similar)
5. **"Weight layouts verified"** — ❌ MoE expert layout was WRONG (interleaved expert data mistaken for contiguous)

### Tensor Dump Verification Checklist
- output.weight: [2048, 248320], type=12 (Q4_K) — OK, access pattern now fixed
- ffn_gate_exps.weight: [2048, 512, 256], type=16 (IQ2_XXS) — INTERLEAVED BUG
- ffn_up_exps.weight: same dims and type
- ffn_down_exps.weight: [512, 2048, 256], type=16 (IQ2_XXS) — SAME BUG
- Shared expert gates: smaller dims, check dequant type
- ssm_beta, ssm_alpha: F32 tensors, no dequant issue

## Next Steps (after fixing MoE expert layout)
1. Fix `dequant_one_expert_contiguous` → `dequant_one_expert_stride`
2. Fix `dequant_multi_expert_contiguous` similarly
3. Rebuild, re-run, compare logits vs reference
4. If still wrong, do layer-by-layer comparison

## Build Command
```bash
cd /home/wubu/bytropix && make infer_text
```
