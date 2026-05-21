# Prestige Prompt — May 21 PM (Phase 29d: Token Embedding Root Cause Found)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**  
**CPU-only: 3-4 tok/s decode (sequential SSM) | GPU vision: 15.7s pipeline**

## Current State
- **CPU text WORKS** with `FORCE_CPU_SSM_SEQ=1`. Coherent output verified.
- **Root cause of 1:1 parity failure found**: Token embedding differs from reference (cs=0.118). `gguf_read_tensor_f32` fails to dequantize quantized `token_embd.weight`. Bytropix embedding has std=0.013 vs reference std=1.295.
- **DUMP_EMBEDDING_DIR** added to `tools/gen_text.c` for embedding comparison.
- All debug infrastructure now built: DUMP_INTERMEDIATE_DIR (ref), DUMP_GQA_DEBUG_DIR (bytropix GQA), DUMP_LAYER_DIR (bytropix layers), DUMP_EMBEDDING_DIR (bytropix embedding).

## Root Cause Analysis (Complete)
1. **Token embedding** (P0.1): cs=0.118 — gguf_read_tensor_f32 dequantization bug ← **THIS IS THE ROOT CAUSE**
2. Secondary: L0-L39 hidden state divergence (cs 0.4→0.18→0.5) — all traced to wrong embedding
3. Both systems output same token (",") despite wrong embedding — the model is numerically robust

## Next Session Priority
1. **Fix token embedding** — either use pre-extracted F32 embeddings or fix gguf_read_tensor_f32
2. **Verify parity** — after fix, embedding should match cs=1.0, then trace L0→L39
3. **Chunked SSM CS>1** — fix FP accumulation

## Key Build Commands
```
make gen_text_cpu
FORCE_CPU_SSM_SEQ=1 DUMP_EMBEDDING_DIR=/tmp/e ./gen_text_cpu "Hello" 1
DUMP_INTERMEDIATE_DIR=/tmp/r ./llama-simple -m /models/... -n 1 "Hello"
```
