═══ GOAL PASTE (May 17 PM v16 — HONEST) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
REF: /home/wubu/llama.cpp/build/bin/llama-cli
STATUS: INFERENCE BROKEN — output `<|endoftext|>Hello_vendor` vs ref "Hello Here's a"

=== FIXED ===
- MoE interleaved dequant FIXED (prior): dequant_block_once + expert_offset = e * D_MODEL * D_FF
- IQ3_XXS block size 98 (not 104) FIXED (prior): MoE down_exps rms 690k→0.25
- IQ4_XS support added (prior): type 23 enum + dequant + raw_size
- IQ1_M dequant FIXED (this session): scale idx ib/4→ib/2, added dl1/dl2 split, removed -1.0f delta shift. Not used in this model.
- Python `dump_gguf.py` type labels CORRECTED (this session): 18→IQ3_XXS, 23→IQ4_XS, added 22→IQ2_S, 29→IQ1_M

=== REMAINING ROOT CAUSE ===
SSM divergence at L0: cos_sim=0.40 vs llama.cpp reference (BEFORE MoE runs).
NOT a dequant issue. In SSM compute path (conv1d, recurrence, output proj, residual).

=== VERIFIED DEQUANTS ===
IQ2_XXS ✅ | IQ2_S ✅ | IQ3_XXS ✅ | Q6_K ✅
IQ4_XS ℹ️ untested | IQ1_M ✅ (unused in this model)

=== ACTUAL TENSOR TYPES (from GGUF) ===
down_exps: IQ3_XXS (37/40 layers) + IQ4_XS (3/40: L34,38,39)
gate/up_exps: IQ2_XXS (all)
gate_inp: F32
shexp gate/up: Q5_K | shexp down: Q6_K
ssm_out: Q6_K | output.weight: Q4_K | token_embd: Q5_K
