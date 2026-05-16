# WuBuText AI — State Dashboard (May 17 PM v16 — HONEST)

## Ground Truth
**INFERENCE STILL BROKEN.** Output: `<|endoftext|>Hello_vendor` (top token 'ore'(9.89)). Reference produces "Hello Here's a".

## DA v9 Findings
- Python `dump_gguf.py` has WRONG type labels: type 18→"IQ2_S" (should be IQ3_XXS), type 23→"IQ1_M" (should be IQ4_XS)
- Actual down_exps: **IQ3_XXS** (type 18) for 37/40 layers, **IQ4_XS** (type 23) for 3/40 (34, 38, 39)
- IQ1_M (type 29) does NOT exist in this model
- IQ3_XXS dequant verified correct vs llama.cpp reference
- IQ2_XXS dequant verified correct
- IQ4_XS dequant written but never tested against reference
- IQ1_M dequant had 3 bugs (scale index, missing dl1/dl2, spurious -1.0f delta) — FIXED but irrelevant for this model

## Verified vs llama.cpp
| Dequant | Status | Notes |
|---------|--------|-------|
| IQ2_XXS | ✅ Correct | Matches reference exactly |
| IQ2_S | ✅ Correct | Matches reference |
| IQ3_XXS | ✅ Correct | Matches reference |
| IQ4_XS | ℹ️ Untested | Written from reference, no cross-dump verified |
| IQ1_M | ✅ Fixed | Was broken (never called in this model) |
| Q5_K | ❓ Previously fixed | High-byte bug fixed in source |
| Q6_K | ✅ Correct | Block layout matches |

## Divergence Point
- **SSM L0 output still diverges at cos_sim ~0.40 vs llama.cpp** (BEFORE MoE)
- This is NOT a dequant issue — it's in the SSM computation
- Possible: Conv1d stack alignment, recurrence step, output projection, residual connection
