# Overnight Map — May 18, 2026 — PHASE 2 COMPLETE

## Session Summary
**Phase 2 done**: Decode 0.6 tok/s (2×). gen_text coherent. DA gaps: 8/10 closed.

## Done
1. Pre-allocated buffers (160→5 mallocs per forward)
2. gen_text: 32-token generation coherent
3. Vault papers read: Qwen3.6 arch, Unsloth UD, DA v10 audit
4. Cos-sim re-verified: 0.9969 (stable, quantization noise)
5. Mind palace updated

## Remaining DA Gaps
- Gap 5: Shared expert sigmoid gate (`ffn_gate_inp_shexp`) not applied in MoE
- Gap 7: No chat template in gen_text (affects quality)

## Next Session
1. Add chat template to gen_text (Gap 7)
2. Fix shared expert sigmoid gate in wubu_moe.c (Gap 5)
3. KV cache for GQA decode (minor speedup)
4. Read Unsloth UD dynamic quant blog for IQ2_M format understanding
