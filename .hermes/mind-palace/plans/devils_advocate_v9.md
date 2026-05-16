# DA v9 — May 17 PM — Quant Type Layer Audit

## Survivorship Bias Strikes Again

**v7/v8 claimed "IQ4_XS fix was addressing non-existent type" — WRONG.** 
v7 was actually CORRECT about IQ4_XS in layers 34/38/39. The Python dump_gguf.py has BAD type labels that misled the analysis.

## Python dump_gguf.py Type Label Errors

The script `tools/dump_gguf.py` maps type IDs to names WRONGLY:
- `type=18 → "IQ2_S"` SHOULD BE `IQ3_XXS` (per llama.cpp `ggml/include/ggml.h`)
- `type=23 → "IQ1_M"` SHOULD BE `IQ4_XS` (per llama.cpp)
- Missing type 22 = IQ2_S
- Missing type 29 = IQ1_M
→ Fix: `dump_gguf.py` GGML_TYPE dict needs correction.

## Actual Quant Types in Qwen3.6-35B-A3B-UD-IQ2_M.gguf

### Expert Tensors (ffn_*_exps.weight)
| Tensor | Quant Type | Layers |
|--------|-----------|--------|
| ffn_down_exps | IQ3_XXS (type 18) | 0-33, 35-37 (37/40 layers) |
| ffn_down_exps | IQ4_XS (type 23) | 34, 38, 39 (3/40 layers) |
| ffn_gate_exps | IQ2_XXS (type 16) | ALL |
| ffn_up_exps | IQ2_XXS (type 16) | ALL |

### Other Key Tensors
| Tensor | Type | 
|--------|------|
| ffn_gate_inp | F32 (type 0) |
| ffn_gate_shexp/up_shexp | Q5_K (type 13) |
| ffn_down_shexp | Q6_K (type 14) |
| attn_gate | Q5_K |
| attn_qkv | Q5_K |
| ssm_out | Q6_K |
| output.weight | Q4_K |
| token_embd | Q5_K |

## Dequant Verification Status

| Type | Status | Verified Against |
|------|--------|-----------------|
| IQ2_XXS (16) | ✅ CORRECT | llama.cpp dequantize_row_iq2_xxs — same formula, grid, signs |
| IQ2_S (22) | ✅ CORRECT | llama.cpp dequantize_row_iq2_s — same block layout, scale formula |
| IQ3_XXS (18) | ✅ CORRECT | llama.cpp dequantize_row_iq3_xxs — same d, qs, scales_and_signs |
| IQ4_XS (23) | ℹ️ UNVERIFIED | Written from reference but never tested against llama.cpp output |
| IQ1_M (29) | ❌ WAS BROKEN → ✅ FIXED | Had spurious -1.0f delta and wrong scale index (ib/4 vs ib/2) |
| Q5_K (13) | ❓ PREVIOUSLY FIXED | High-byte indexing was wrong, fixed in source |
| Q6_K (14) | ✅ CORRECT | Matches llama.cpp block layout |
| Q4_K (12) | ✅ CORRECT | Matches llama.cpp |

## IQ1_M Bug Details (Fixed This Session)

The `dequantize_iq1_m_row` function had THREE bugs:
1. **Scale index**: `sc_idx = ib / 4` only read 2 of 4 scale words (sc[0], sc[1]). Should be `ib / 2` to read all 4.
2. **Missing dl1/dl2**: Only 1 scale per 32-element sub-block. llama.cpp ref has TWO (dl1 for first 16, dl2 for last 16), split at 3-bit boundaries.
3. **Spurious -1.0f delta**: Delta was `-1.0f ± IQ1M_DELTA` instead of `±IQ1S_DELTA`. Shifted all dequantized values by -1.0.

→ Fixed to match llama.cpp `dequantize_row_iq1_m()` exactly.

**BUT**: Iq1_M (type 29) does NOT appear in the UD-IQ2_M model. Fix is correct but irrelevant for current test.

## Remaining Issue

SSM L0 cos_sim = 0.40 vs reference (BEFORE MoE). This is NOT a dequant issue — it's in the SSM computation itself. Output projection? Conv1d? Recurrence? Post-SSM residual connection?

## Action Items
1. Fix `dump_gguf.py` type labels
2. Verify IQ4_XS dequant against llama.cpp by cross-dump comparison
3. Write cos_sim comparison test for MoE tensor outputs vs llama.cpp
