# WuBuText AI — State Dashboard (May 16 PM v12 — HONEST)

## Ground Truth
**infer_text_gpu v5 does NOT produce correct output.** 
`"The capital of France is"` → our output: `iscInset了下去idesiby客的我们都会论usher...`
llama.cpp output: `Here's a thinking process:\n\n1.`

All ✅ statuses below mean "compiles and doesn't crash" unless otherwise noted.

---

## Inference Engines — Real Status

| Binary | Claimed | Actual Status | Real Notes |
|--------|---------|---------------|------------|
| `infer_text_gpu v5` | ✅ 245 tok/s | ❌ Garbage output | Speed real but output wrong. SGEMM ldC bug FIXED. RoPE added. Sampling added. |
| `infer_text v2` | ✅ | ❌ Garbage output | Same bugs as GPU. RoPE added to prefill (May 16). Decode path RoPE still missing? |
| `train_integrated` | ✅ CE 12.42 | ❓ CE measured | No reference baseline. Could be learning garbage patterns. |
| `infer_moe_lazy` | ✅ 37 tok/s | ❓ Speed measured | Speed claim verifiable. MoE output never checked for correctness. |
| `infer_unified` | ✅ | ❌ Garbage | Same architecture as infer_text, same bugs. |
| `test_kv_cache` | ✅ max_diff=0.00 | ✅ Verified | Numerical comparison vs recompute — gold standard test. |
| `test_256k` | ✅ MoE router 65K | ✅ Individual component | Only tests MoE router, not full pipeline. |
| `infer_vision_gpu` | ✅ 99ms | ❓ Moondream3 | Separate model (not Qwen3.6). 27 GPU layers, 0 NaN. |

## Fixed Bugs (May 15-16)

| Bug | Fix | Verified? |
|-----|-----|-----------|
| SGEMM ldC=vocab_size (was writing logits to wrong addresses) | ✅ ldC=N | ❓ Logits non-zero now, but output still wrong |
| Q5_K dequant high-bit byte indexing | ✅ New qh_v1_base/v2_base logic | ❓ Old and new code produce same float values for this data |
| RoPE missing from CPU GQA prefill | ✅ Added sin/cos table + apply_rotary | ❓ Output unchanged after fix |
| BOS not prepended | ✅ pids[0]=bos_id | ❓ Output still wrong (just has <|endoftext|> prefix now) |
| Temperature/top-k/top-p sampling | ✅ sample() function added | ❓ Greedy still produces garbage |
| EOS detection (eos=bos=248044) | ✅ gen>2 threshold | ❓ Never reaches EOS — output loops on garbage tokens |

## Known Broken (No Fix Yet)

| Problem | Likely Root Cause | Evidence |
|---------|------------------|----------|
| MOE=0 = no FFN at all | Model is MoE-only (no dense FFN layers) | Config: `moe_intermediate_size: 512`, no `intermediate_size` |
| MOE=1 also produces garbage | MoE dequant (IQ2_XXS/IQ3_XXS), SSM fwd, or tokenizer | Output completely different from llama.cpp reference |
| CPU vs GPU output diverges | Different architectures (SSM impl, RoPE, gate application) | Different first token from same prompt |
| No inference binary has been verified against reference | No golden outputs for anything except "life!!!" | Test suite passes but only checks compilation + non-crash |
| Tokenizer is custom, not GGUF-native | Pre-extracted vocab/merges files from who-knows-when | No comparison with llama.cpp's tokenizer output |

## llama.cpp Reference (May 16)

Built at `~/llama.cpp/build/bin/llama-cli`
- Output for "The capital of France is": `Here's a thinking process:\n\n1.` 
- GGUF-native tokenizer (correct for this model)
- Uses full llama.cpp architecture (not our custom SSM/GQA impl)
- **Our reference standard for all future comparisons**

## Vault Versioning

Previous mind palace versions archived to `vault/bins/`:
- state-v10-May15-PM.md
- plan-v11-May16-AM.md
- goal-mantra-v10-May15-PM.md
- testing-v7-May15-PM.md
- project-May15.md
- entry-v6-May15-PM.md
- mind-palace-README-v6.md
- index-v6-May15-PM.md
- overnight-map-v6-May15-PM.md
- STATUS-v3-May15-PM.md

New versions overwrite .hermes/mind-palace/*.md. Old versions preserved in vault/bins/.

## TGT Math
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)

## Actual Priority
P0 — Fix inference (compare vs llama.cpp layer by layer)
P1 — Verify all existing component tests against reference
P2 — Fill in hyperbolic backward passes (forward-only = can't train)
P3 — GPU acceleration, tailslayer, 256K scaling
