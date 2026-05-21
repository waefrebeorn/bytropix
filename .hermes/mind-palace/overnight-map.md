# Overnight Map — May 21 PM (Phase 29d: Token Embedding Root Cause Found)

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 4ebe712 (uncommitted: DUMP_EMBEDDING_DIR in gen_text.c)  
**Model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, only model)  
**Reference:** /home/wubu/llama.cpp/build/bin/llama-simple (fast, no chat template)

## Session Summary (Phase 29d)

### Root Cause Found: Token Embedding
**The bytropix token embedding differs from reference with cs=0.118.** 
- Reference `global_model.input_embed.bin`: mean=0.011, std=1.295 (correct)
- Bytropix `embedding.bin`: mean=3.1e-5, std=0.013 (nearly zero)
- `gguf_read_tensor_f32()` fails to dequantize quantized `token_embd.weight`
- This explains the L0 cs=0.405 and all downstream divergence up to L39

### New Debug Infrastructure
| Feature | Files Changed | Description |
|---------|--------------|-------------|
| DUMP_EMBEDDING_DIR | `tools/gen_text.c` | Dumps embedding buffer after token lookup |

### Verified Claims
| Claim | Status | Evidence |
|-------|--------|----------|
| Token embedding broken | ✅ | cs=0.118 vs reference. Bytropix std=0.013, ref std=1.295 |
| Root cause found | ✅ | gguf_read_tensor_f32 dequantization bug |
| L0 cs=0.405 explained | ✅ | Direct consequence of wrong embedding |
| All downstream divergence explained | ✅ | Wrong input propagates through 40 layers |

## Current Blockers
1. **Token embedding dequantization**: gguf_read_tensor_f32 doesn't handle quantized token_embd.weight
2. **Quick fix path**: Pre-extracted F32 embeddings at `data/qwen36_embeddings_c.bin.raw` may work
3. **Chunked SSM CS>1**: Must use `FORCE_CPU_SSM_SEQ=1`

## Workstreams (next session)
**A — Fix token embedding**: Either use pre-extracted F32 file or fix gguf_read_tensor_f32

**B — Verify full parity**: After embedding fix, compare L0 and trace forward

**C — Chunked SSM**: Only after parity confirmed

## Data NOT to Re-Derive
- Token embedding cs=0.118 vs reference (not 1.0)
- L0 divergence cs=0.405 is a SECONDARY effect of wrong embedding
- L31 GQA was never the root cause
- Both systems output same token (",") despite wrong embedding — model is tolerant
