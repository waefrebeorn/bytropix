# Training Backward Integration — Findings

## Root Cause of Backward Crash

The SSM backward function (`wubu_ssm_backward`) crashes because it requires F32 weight pointers (`w->ssm_out_weight`, `w->attn_output_weight`, etc.) which are not loaded by `wubu_model_init`.

### Current State

| Weight | F32 ptr | Quant ptr | Forward uses | Backward needs |
|--------|---------|-----------|--------------|----------------|
| ssm_out_weight | NULL | valid `_q` + `_type` | quantized path | F32 ptr (crashes) |
| attn_qkv_weight | NULL | valid | quantized path | F32 ptr |
| attn_gate_weight | NULL | valid | quantized path | F32 ptr |
| ssm_beta_weight | ✅ F32 | — | F32 | F32 ✅ |
| ssm_alpha_weight | ✅ F32 | — | F32 | F32 ✅ |

### Options

**Option A: Dequantize on backward** — In `wubu_ssm_backward`, detect NULL F32 pointers and dequantize from the Q variant on-the-fly. Memory-efficient but adds ~200ms per layer for dequant.

**Option B: Load F32 weights during init** — Add F32 weight loading for output weights. Memory cost: ssm_out_weight = 2048*2048*4 = 16MB per layer × 30 = 480MB. Plus GQA equivalents.

**Option C: Hybrid approach** — Load output weights as F32 (they're needed for gradient), keep attention weights quantized (dequant inside backward if needed).

### Next Steps for Training

1. Minimal fix: load `ssm_out_weight` as F32 during model init (Option B for key weights)
2. Then re-run backward validation test
3. Then wire per-layer backward into train_real.c
