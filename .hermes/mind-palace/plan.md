# Plan — May 21 PM (Phase 29d: Token Embedding Root Cause Found)

## P0: 1:1 Parity with llama.cpp — Corrected Approach

The root cause of 1:1 parity failure has been identified: **token embedding** differs between bytropix and llama.cpp (cs=0.118). This explains the L0 divergence cs=0.405, not any GQA or SSM computation bug.

### P0.1: Fix Token Embedding (Highest Priority)
The `token_embd.weight` tensor in the GGUF file is stored in a quantized format. `gguf_read_tensor_f32()` doesn't properly dequantize it, producing near-zero values (std=0.013 vs expected std=1.295).

**Two approaches:**
1. **Quick fix** — Use pre-extracted F32 embeddings from `data/qwen36_embeddings_c.bin.raw`. Check if this file exists and has correct values. Set `use_embedding_file=true` via env var.
2. **Proper fix** — Fix `gguf_read_tensor_f32()` to handle the quantization type of `token_embd.weight`. Check what quantization type is used and add dequantization.

**Action:**
1. Check if `data/qwen36_embeddings_c.bin.raw` exists and has correct values
2. If yes, enable `use_embedding_file` and verify cs=1.0 for embedding
3. If no, fix `gguf_read_tensor_f32` to dequantize properly

### P0.2: Verify Full Parity After Fix
Once token embedding matches cs=1.0:
1. Compare L0 output (cs should approach 1.0)
2. Trace forward layer by layer
3. Identify remaining divergence sources (SSM accumulation, quantization)

### P0.3: L31 GQA and Other Layer Debugging (De-prioritized)
These were secondary effects. Fix embedding first.

## P1: Structural Fixes
### P1.1: Chunked SSM CS>1
Still broken. Workaround: `FORCE_CPU_SSM_SEQ=1`.

### P1.2: gen_text binary naming
✅ **DONE** — symlink `gen_text → gen_text_cpu`.

## Root Cause Evidence
| Metric | Reference | Bytropix |
|--------|-----------|----------|
| Embedding mean | 0.011 | 3.1e-5 |
| Embedding std | 1.295 | 0.013 |
| Embedding cs | — | 0.118 |
| L0 hidden cs | — | 0.405 |
| L30 hidden cs | — | 0.182 |
| L31 hidden cs | — | 0.471 |

The embedding is nearly zero but not all-zero, suggesting quantized values are partially loaded but not properly dequantized.
