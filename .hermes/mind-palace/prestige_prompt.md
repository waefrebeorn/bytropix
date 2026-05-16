═══ WUBUTEXT AI — PRESTIGE RESUME (May 17 PM v17 — HONEST) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120
Build: make infer_text | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE (May 17 PM v17) ===
✅ P0: IQ3_XXS block size 104→98 FIXED (prior session)
✅ P0: IQ4_XS enum + dequant + raw_size added (prior session)
✅ P0: MoE interleaved dequant FIXED (prior session)
✅ DA v9: Python dump type labels BAD → CORRECTED
✅ IQ1_M dequant FIXED (had -1.0f delta shift, wrong scale index, missing dl1/dl2)
   - NOT used in this model (no type 29 tensors). Future-proof fix.
▶ P0: SSM divergence — L0 cos_sim=0.40 vs reference (BEFORE MoE)

=== DA v9 KEY FINDING ===
Python `tools/dump_gguf.py` had WRONG type labels:
- type=18 was "IQ2_S" → ACTUALLY IQ3_XXS
- type=23 was "IQ1_M" → ACTUALLY IQ4_XS
- Missing: type 22 (IQ2_S), type 29 (IQ1_M)
Actual down_exps types: IQ3_XXS (37 layers), IQ4_XS (3 layers: 34,38,39)
v7/v8 analysis was misled by bad labels → v7 IQ4_XS fix was CORRECT

=== VERIFIED DEQUANTS ===
IQ2_XXS ✅ | IQ2_S ✅ | IQ3_XXS ✅ | Q6_K ✅ | Q5_K (previously fixed) ❓
IQ4_XS ℹ️ (untested) | IQ1_M ✅ (irrelevant)

=== REMAINING ===
P0 — SSM output wrong: cos_sim 0.40 at L0 before MoE. Not dequant issue.
P0 — Output: `<|endoftext|>Hello_vendor` vs ref "Hello Here's a"
