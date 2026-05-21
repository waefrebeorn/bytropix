# Goal Mantra — May 21 PM (Phase 29d: Token Embedding Root Cause Found)

**Target:** DUMP_EMBEDDING_DIR added. Root cause identified: gguf_read_tensor_f32 produces wrong token embeddings.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| CPU text (sequential) | ✅ 3-4 tok/s | FORCE_CPU_SSM_SEQ=1. Coherent. |
| CPU text (chunked CS>1) | ❌ Garbled | FP accumulation across 30 layers |
| DUMP_EMBEDDING_DIR | ✅ Built | Dumps token embedding for 1:1 comparison |
| Root cause found | ✅ | Token embedding cs=0.118. gguf_read_tensor_f32 bug. |
| Llama.cpp DUMP_INTERMEDIATE_DIR | ✅ Rebuilt | libllama.so + llama-simple |
| Bytropix DUMP_GQA_DEBUG_DIR | ✅ Built | Per-layer GQA intermediate dumps |
| gen_text symlink | ✅ DONE | gen_text → gen_text_cpu |

## P0: Corrected Priority
1. **Fix gguf_read_tensor_f32** for quantized token_embd.weight — need to handle quantization type in GGUF tensor load
2. **Alternative**: Load pre-extracted F32 embeddings from `data/qwen36_embeddings_c.bin.raw` (already exists)
3. **Verify**: After fix, cs should be 1.0 for embedding, then re-check L0-L39

## Key Finding
Token embedding cs=0.118 vs reference. gguf_read_tensor_f32 doesn't handle the quantized format. Reference `global_model.input_embed.bin` has std=1.295, bytropix has std=0.013 (nearly zero). This explains the L0 cs=0.405 and all downstream divergence.

## BUILD
```
make gen_text_cpu
FORCE_CPU_SSM_SEQ=1 DUMP_EMBEDDING_DIR=/tmp/e ./gen_text_cpu "Hello" 1
DUMP_INTERMEDIATE_DIR=/tmp/r ./llama-simple -m /models/... -n 1 "Hello"
```
