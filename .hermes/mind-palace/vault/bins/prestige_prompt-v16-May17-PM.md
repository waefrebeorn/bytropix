═══ WUBUTEXT AI — PRESTIGE RESUME (May 17 PM v16 — HONEST) ═══
Path: /home/wubu/bytropix | Branch: master (bceb160)
HW: RTX 5050 6.4GB, -arch=sm_120
Build: make infer_text | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE (May 17 PM v16) ===
✅ P0: IQ3_XXS block size 104→98 FIXED
   root cause of MoE down_exps garbage (rms 690k→0.25)
   dequant stride was 6 bytes past each block → progressively corrupt
✅ P0: IQ4_XS support added (type 23 enum, raw_size=136, dequant function)
   was mapped to IQ1_M=23 (wrong), IQ1_M=29 actual. Layers 34/38/39.
✅ P0: MoE interleaved dequant FIXED (prior session)
✅ Shared expert output reasonable (rms=0.51)
✅ All 40 layers process without crash (135s prefill)
▶ P0: SSM divergence — NOT IMRoPE (SSM doesn't use RoPE)
   SSM cos_sim=0.40 at L0 vs reference (same before/after MoE fix)
   "Python verified" claim was survivorship bias (only Python=C consistency)
   NEED: compare SSM outputs against llama.cpp directly

=== REMAINING ===
P0 — SSM output wrong: compare SSM formula + weights vs llama.cpp
P0 — Model config values hardcoded, not verified from GGUF metadata
P1 — IQ4_XS dequant unverified against real IQ4_XS tensor data
P1 — IQ3_XXS internal format may differ from llama.cpp (qs[64]+scales[32] vs qs[96])

=== DA v8 FINDINGS ===
- "IMRoPE is root cause" (DA v3/v7) → CONFIRMED STALE. SSM doesn't use RoPE.
- "SSM verified" → survivorship bias. Only Python=C matching, not correctness.
- IQ4_XS dequant → written from source, never run on actual tensor.
- Model config → all hardcoded, not cross-checked vs GGUF metadata.
