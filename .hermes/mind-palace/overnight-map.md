# WuBuText AI — Overnight Navigation Map (May 16 v15 — HONEST)

## Where We Are
All P0-P3 infrastructure tasks completed this session. P1a (chunked DeltaNet) implemented but is a training-only path — NOT used in inference. Current inference uses sequential recurrence (wubu_ssm_forward), which is correct.

Remaining "Doug" vs "Here" discrepancy — attn_output_gate is NOT the cause (confirmed implemented). Root cause still unknown.

## What's Done This Session
1. **P1a — Chunked DeltaNet**: Full implementation matching delta-net-base.cpp
   - Fixed triangular solve direction (bottom-up for L^T)
   - Fixed decay mask triangle (lower tri, was upper)
   - Fixed KQ to use LOWER_DIAG
   - Note: chunked≠sequential for multi-token — expected, not a bug
2. **P2a, P3a**: Verified clean rebuild (done prior session)
3. **Model type audit**: Confirmed all 7 types used by model are supported
4. **Unsloth Dynamic 2.0 research**: Understanding complete

## Model Research
- 40 layers, 10x(3x SSM→MoE→1x GQA→MoE). 
- Mixed quant: attention Q5_K, MoE experts IQ2_XXS/IQ3_XXS/IQ4_XS, SSM out Q6_K
- "IQ2_M" = Unsloth Dynamic multi-level label, not actual type

## Remaining Items
- "Doug" vs "Here": investigate tokenizer BOS handling, embd_norm epsilon
- Possible: quantization noise floor at 2-3 bpw on 35B MoE
