═══ GOAL PASTE (May 16 v25 — DA RECERTIFIED) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (7-type mixed quant)
STATUS: All inference pipeline STRUCTURE verified correct. Output WRONG.

=== DA RECERT (all prior ✅ stripped, re-audited) ===
❌ "Hello" → "Plot" — root cause UNKNOWN, 5 suspects remain.
❌ All P0-P3 completed at "compiles" level only — ZERO verified against llama.
❌ Auto-embedding depends on dequant correctness of type 0 (F32) — safe.
✓ BOS: ADD_BOS off matches add_bos_token=false.
✓ Weight indexing: infer_text.c uses correct i + j*D_MODEL pattern everywhere.
✓ RoPE applied in both prefill + decode paths.
✓ SSM recurrence structure matches delta-net-base.cpp.
✓ MoE router + shared expert structure correct.

=== 5 ACTIVE SUSPECTS (ranked) ===
1. Q5_K dequant (181 tensors) — high impact, easy to verify with test vector
2. Output weight type 12 Q4_K dequant — corrupts logits directly
3. SSM Q scaling 1/sqrt(128) — verify llama.cpp applies same
4. RMSNorm epsilon (1e-6 vs llama.cpp)
5. TGT wrapping in GQA scores — clips to [-π,π], not in llama.cpp

=== CRITICAL FIXES NEEDED ===
- wubu_gqa_forward() weight indexing: uses i*cols+j but should use i+j*D_MODEL
  (Dead code for inference — only affects deprecated wubu_model_forward_from_embd API)

=== NEXT ===
Write Q5_K dequant test vector. Compare single-block f32 values vs llama.cpp.
Then output weight dequant test.
