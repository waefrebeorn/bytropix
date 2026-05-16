=== WuBuText AI — GOAL PASTE (May 17 PM v15 — HONEST) ===

HARD TRUTH: Inference still broken. 2 more bugs fixed this session but SSM root cause remains.

=== FIXED THIS SESSION ===
- ✅ IQ3_XXS block size: 104→98 bytes (DEQUANT STRIDE BUG — was the root of MoE garbage)
- ✅ IQ4_XS support: enum, raw_size(136), dequant function added (layers 34/38/39)
- ✅ MoE output now REASONABLE: rms=0.25, max=2.2 (was rms=690k, max=4.4M)

=== OUTPUT STATUS ===
- Reference (llama.cpp): Correct — Hello, Here's a thinking process...
- Our infer_text (40L, MOE=1): Top token "的发展和" (11.68), output "Hello发展壮大"
- SSM-only (MOE=0): Still wrong — bug is in SSM, not MoE

=== WHAT WORKS ===
- All 40 layers process without crash ✅
- MoE expert dequant (IQ2_XXS gate/up, IQ3_XXS down, IQ4_XS down) ✅
- MoE output now reasonable rms=0.25 ✅
- SSM QKV/conv/recurrence verified vs Python (cos_sim=1.0) ✅
- Shared expert output reasonable (rms=0.51) ✅

=== WHAT'S BROKEN ===
- SSM L0 cos_sim=0.40 vs reference (same as before these fixes)
- Bug location: between SSM gated value and residual — either output projection or SSM weight loading
- Full 40-layer h_last cos_sim=0.004

=== NEXT ===
P0 — Compare SSM value output (before output projection) against reference
P1 — Verify SSM output weight (ssm_out.weight) dequant and shape

=== COMMITS ===
bceb160 — fix(gguf): IQ3_XXS block size 104->98, add IQ4_XS dequant

REFERENCE: ~/llama.cpp/build/bin/llama-cli
BUILD: make infer_text
