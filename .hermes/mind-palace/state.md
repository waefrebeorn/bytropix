# WuBuText AI — State Dashboard (May 16 PM v13 — HONEST)

## Ground Truth
**ALL inference binaries produce garbage output. Q5_K dequant fixed but not root cause.**
Prompt "The capital of France is": 
- Our output (MOE=0): `ò` then loops `_tuples` → `ò` → `ò`...
- Our output (MOE=1): `新项目` then jumps between languages: `judging'i quanto_pass...`
- **llama.cpp reference**: `Here's a thinking process:\n\n1.`

All ✅ statuses mean "compiles and doesn't crash" unless noted.

---

## Inference Engines — Real Status

| Binary | Claimed | Actual | Notes |
|--------|---------|--------|-------|
| `infer_text` v2 | ✅ | ❌ Garbage | Q5_K dequant fix applied. `ò`→`_tuples` loop (MOE=0) or lang-jumping (MOE=1) |
| `infer_text_gpu` v5 | ✅ 245 tok/s | ❌ Garbage | Same bugs as CPU + GPU output projection |
| `train_integrated` | ✅ CE 12.42 | ❓ CE measured | No reference baseline |
| `infer_moe_lazy` | ✅ 37 tok/s | ❓ Speed only | Output never verified |
| `test_kv_cache` | ✅ max_diff=0.00 | ✅ Verified | Gold standard |
| `test_256k` | ✅ MoE router 65K | ✅ Verified | Component only |
| `infer_vision_gpu` | ✅ 99ms | ❓ Moondream3 | Separate model |

## Fixed Bugs (May 16 PM)

| Bug | Fix | Status |
|-----|-----|--------|
| Q5_K dequant qh bit-indexing | ✅ Corrected to match llama.cpp reference (qh[l] bit[chunk_id*2+0/1]) | Output changed, still garbage |
| TGT state wrap removed | ✅ Removed from SSM forward (not in llama.cpp ref) | No effect on short sequences |
| GQA gate verification | ✅ Already applied in both prefill + decode paths | Was never a bug |
| EOS detection | ✅ gen>1 check for eos=bos=248044 | Correct |
| Debug logit dump | ✅ Added top-5 prefill + step-by-step decode | Diagnostic only |

## Known Broken (No Fix Yet)

| Problem | Evidence |
|---------|----------|
| MOE=0: loops `ò` (21502) and `_tuples` (86196) | Logits flat (9-11 range for top-5). SSM output mean=2.85 vs embedding mean=0.02 (140×) |
| MOE=1: jumps languages each token | Different tokens each step but semantically garbage |
| Causality likely SSM weight scaling | SSM output projection (`ssm_out.weight`) Q5_K dequant may still be wrong, or SSM recurrence formula differs from Qwen3.6 |

## Hidden State Diagnostics

| Metric | Value | Healthy? |
|--------|-------|----------|
| Embedding mean | 0.02 | ✅ Normal |
| Embedding max | 0.20 | ✅ Normal |
| Layer 0 SSM attn_out mean | 2.85 | ❌ 140× embedding magnitude |
| Layer 0 SSM attn_out max | 47.9 | ❌ Extreme outlier |
| Output logits top-5 range | 8.86-10.04 | ❌ Flat (gap <0.5 between positions) |

## What's Verified Working
- **test_kv_cache**: max_diff=0.00 ✅
- **test_256k**: MoE router O(T) to 65K ✅
- **Q4_K dequant**: Matches llama.cpp reference ✅
- **Q5_K dequant**: Fix aligns with llama.cpp reference ✅ (but output still wrong)
- **llama.cpp reference**: BUILT at ~/llama.cpp/build/bin/llama-cli ✅
- **API server sandbox**: 14 tests pass ✅
- **GQA gate**: Applied correctly in both paths ✅
- **EOS detection**: Correct logic ✅

## Commits This Session
- `39aeaa1` — Q5_K dequant qh fix + debug logit dump
- `4e8a216` — TGT wrap removed from SSM forward
- `ba4b43b` — Hidden state magnitude debug dump

## Priorities
P0 — Fix inference: root cause in SSM weight scaling or dequant of other types
P1 — Verify layer-by-layer hidden states vs llama.cpp
P2 — Fix training hyperbolic backward passes
P3 — GPU acceleration, tailslayer, 256K
