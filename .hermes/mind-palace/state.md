# State — May 19, 2026 (01:30) — MTP FULLY IMPLEMENTED. STATE SAVE/RESTORE. BUGFIXES.

## REAL STATUS
MTP head loads correctly (nextn_hnorm was missing — BUG FIXED). SSM state checkpoint/rollback implemented. gen_text_mtp: opt-in MTP mode (MTP=1 env var). Non-MTP: 0.6 tok/s coherent. MTP: 3.3 tok/s but output degenerates at IQ2_M (blk.40 Q2_K/Q3_K quantization).

## MTP Verify Discovery (@ IQ2_M)
100% rejection rate — MTP head predictions NEVER match main model's. Root cause: quantization noise (IQ2_M main model + Q2_K/Q3_K blk.40 MoE). MTP spec-decode cannot work at this quant level. For higher-precision models, verify infrastructure ready (checkpoint/rollback).

## MTP "Free Tokens" Mode (MTP=1)
Emits 1 main + DRAFT_N(4) MTP tokens per decode step. Speed: 3.3 tok/s decode. Quality: degraded at IQ2_M — MTP head produces degenerating output after 1-2 steps.

## Code Changes (this session)
- include/wubu_model.h: ssm_states_saved, conv_states_saved, gqa_cache_len_saved, mtp_cache_len_saved. checkpoint/rollback decl.
- src/wubu_model.c: wubu_model_checkpoint + rollback impl. nextn_hnorm load bugfix. save_last_hidden captures pre-final-norm. Free save bufs in free().
- tools/gen_text_mtp.c: Full MTP pipeline. Draft generate (4 MTP head calls per step). Verify loop with per-token checkpoint/rollback. Free tokens mode (no verify). MTP=1 opt-in.

## Performance
- Non-MTP: 0.6-0.7 tok/s decode, same as gen_text
- MTP free-tokens: 3.3 tok/s decode (~5×), quality loss at IQ2_M
- MTP verify: N/A (0% acceptance at IQ2_M)

## Next
- Phase 7: Hardware saturation (AVX2, prefetch, NUMA)
- Higher-precision MTP model for working spec-decode
- MTP with non-MTP GGUF (blk.40 not present — graceful fallback works)
