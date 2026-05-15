=== WuBuText AI — GOAL PASTE (May 16 v12 — HONEST) ===

HARD TRUTH: infer_text_gpu v5 compiles and runs fast but produces GARBAGE.
All "✅" on binaries means "doesn't crash" NOT "produces correct output."
Reference (llama.cpp): "Here's a thinking process:" — Us: "iscInset了下去idesiby客的我们都会论usher..."

=== WHAT WE ACTUALLY HAVE ===
- infer_text_gpu v5: GPU accelerated, fast decode, but WRONG output
- infer_text v2: CPU baseline, also WRONG output
- llama.cpp: BUILT at ~/llama.cpp — reference for all comparisons
- Q5_K dequant: Suspected bug, fix in source but impact unconfirmed
- API server: tools/serve.py works in sandbox, needs working inference backend
- SSM/GQA kernels: Exist and run, but produce wrong results
- all 6 training flags: WIRED but most are forward-only (no gradient flow)
- vault: 60+ research docs, architecture references, paper summaries

=== COMPLETED (Verified Working) ===
- test_kv_cache: KV cache matches full recompute (max_diff=0.00) ✅
- test_256k: MoE router O(T) scaling verified to 65K ✅
- Q5_K dequant: Source fix applied ✅
- RoPE: Added to CPU GQA prefill path ✅
- Sampling: temp/top-k/top-p added ✅
- API server: Built + sandbox tested (14 tests) ✅
- EOS detection: Fixed (eos=bos=248044) ✅
- SGEMM ldC bug: Fixed (zero-logits problem) ✅

=== P0 — FIX INFERENCE (Everything depends on this) ===
- Compare hidden states layer-by-layer vs llama.cpp
- Fix SSM forward pass (most likely root cause)
- Fix MoE dequant (IQ2_XXS/IQ3_XXS)
- Fix tokenizer or switch to GGUF-native
- Must produce "Here's a thinking process:" not garbage

=== P1 — VERIFY ALL COMPONENTS ===
- train_integrated: What does CE=12.42 actually mean?
- Poincaré graft: Verify 95% NN preservation against raw embeddings
- Vision pipeline: Works for Moondream3 but is that model correct?
- All "✅" in old state.md: Re-verify against reference

=== P2 — HYPERBOLIC BACKWARD PASSES ===
- Poincaré GQA backward (missing — forward only)
- Nested SSM backward (missing — forward only)
- Nested MoE backward (missing — forward only)

=== P3 — GPU ACCELERATION & SCALE ===
- MoE on-device dequant (PCIe bottleneck: 21GB/token)
- Tailslayer spec decode
- 256K context verification

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]
Useful only when inference works.

BUILD: make infer_text_gpu | REFERENCE: ~/llama.cpp/build/bin/llama-cli
HW: RTX 5050, sm=120

Old versions: vault/bins/state-v10-May15-PM.md, plan-v11-May16-AM.md, goal-mantra-v10-May15-PM.md
