=== WuBuText AI — GOAL PASTE (May 16 v13 — HONEST) ===

HARD TRUTH: ALL inference binaries produce garbage.
3 bugs fixed this session (Q5_K dequant, TGT wrap, GQA gate verified) but root cause remains.
Output (MOE=0): loops `ò`(21502)/`_tuples`(86196) — logits flat 9-11.
Output (MOE=1): jumps languages each step — semantically dead.
Reference (llama.cpp): "Here's a thinking process:\n\n1."

=== WHAT ACTUALLY WORKS ===
- llama.cpp: BUILT at ~/llama.cpp/build/bin/llama-cli — correct output ✅
- test_kv_cache: KV cache matches full recompute (max_diff=0.00) ✅
- test_256k: MoE router O(T) scaling to 65K ✅
- Q4_K dequant: verified vs llama.cpp reference ✅
- Q5_K dequant: qh bit-indexing FIXED (matches ref) but not root cause
- GQA gate: applied in both prefill + decode paths ✅
- API server: tools/serve.py sandbox (14 tests) ✅
- EOS detection: correct (gen>1 for eos=bos=248044) ✅
- NaN in training: FIXED (MoE weight interleaving) ✅
- Per-expert dequant: 11s/step (16×), 0 NaN all configs ✅

=== WHAT'S BROKEN (P0) ===
- ALL inference binaries produce garbage
- SSM output mean=2.85 vs embedding mean=0.02 (140× magnitude)
- Root cause: SSM weight dequant (ssm_out.weight Q5_K) or SSM recurrence formula
- MOE=0: loops `ò`/`_tuples` (stuck in 2-token attractor)
- MOE=1: jumps random languages (dequant errors in IQ2_XXS/IQ2_S?)
- Q5_K dequant fix changed output but didn't fix it
- TGT wrap removal had no effect on 6-tok sequences

=== DIAGNOSTICS ===
Embedding stats: mean=0.02 max=0.20 — normal
Layer 0 SSM attn_out: mean=2.85 max=47.9 — 140× embedding magnitude (suspicious)
Top-5 logits: 8.86-10.04 range — flat (gap <0.5)

=== PRIORITY ===
P0 — Fix inference by verifying weights layer-by-layer vs llama.cpp
P1 — Verify all components against reference
P2 — Hyperbolic backward passes (forward-only = can't train)
P3 — GPU acceleration, tailslayer, 256K

=== COMMITS THIS SESSION ===
39aeaa1 — Q5_K dequant qh fix + debug logit dump
4e8a216 — TGT wrap removed from SSM forward
ba4b43b — Hidden state magnitude debug dump

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]
REFERENCE: ~/llama.cpp/build/bin/llama-cli
BUILD: make infer_text
HW: RTX 5050, sm=120
