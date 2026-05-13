# Phase 1: Embedding Grafting ✅ DONE (May 12)

**Goal:** Extract Qwen3.6-35B-A3B token embeddings → map to Poincaré ball → verify quality.

**Status:** ✅ Complete. All steps verified.

## What Was Done

### Step 1.1-1.3: GGUF Reader + Extraction
- **C code**: `include/gguf_reader.h`, `src/gguf_reader.c` — full GGUF parser, Q5_K dequant, Poincaré exp/log maps
- **Python**: `tools/extract_embeddings.py` — verified identical results to C
- **Python**: `tools/analyze_embeddings.py` — norm distribution + NN preservation test
- **Output files**:
  - `data/qwen36_embeddings_c.bin` — Poincaré-mapped (R=0.956, 2.03GB)
  - `data/qwen36_embeddings_c.bin.raw` — raw dequantized (2.03GB)
  - `data/qwen36_embeddings_c.bin.meta` — shape and stats

### Key Discoveries
- **token_embd.weight is Q5_K** (type 13, not IQ2), despite filename "IQ2_M"
  - Different tensors use different quantization types in the same GGUF
  - Embeddings kept at higher precision (5-bit) than expert weights (IQ1_S/IQ2_XS)
- **97.3% NN preservation** at R=0.956 (3 × mean_norm)
  - R=0.5 through R=2.0 all gave ≥94% — very robust to R choice
- **73 zero-norm embeddings** found (padded special tokens at scattered indices)
  - These map to origin in Poincaré ball — handled by exp_map norm check
- **Norm range**: 0.0098 to 0.547 (non-zero), very tight distribution

### Files Created
```
src/gguf_reader.c              — GGUF parsing, Q5_K dequant, exp/log maps
include/gguf_reader.h          — Public API
tools/extract_embeddings.py    — Python GGUF reader (for verification)
tools/extract_and_map.c        — CLI: extract + map to Poincaré
tools/analyze_embeddings.py    — Distribution + NN preservation analysis
tools/dump_gguf.py             — GGUF structure dumper
data/qwen36_embeddings_c.bin   — Mapped embeddings (input for Phase 2+)
```

### Pitfalls Resolved
- ~~GGUF quantization (IQ2_M = 2-bit)~~ → Actual type is Q5_K for embeddings
- ~~vocab_size=248320 is padded → last tokens might be all zeros~~ → 73 scattered zeros, not contiguous
- ~~Make sure we extract text embeddings, not vision~~ → Verified: only token_embd.weight extracted

