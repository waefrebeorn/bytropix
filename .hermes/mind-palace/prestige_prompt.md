# Prestige Prompt — May 21 PM (Phase 29c: DUMP_INTERMEDIATE_DIR Hooks + Divergence Audit)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**  
**CPU-only: 3-4 tok/s decode (sequential SSM) | GPU vision: 15.7s pipeline**

## Current State
- **CPU text WORKS** with `FORCE_CPU_SSM_SEQ=1`. Coherent output verified.
- **Llama.cpp inline hooks DONE** — `DUMP_INTERMEDIATE_DIR` works in rebuilt `llama-simple` + `libllama.so`. Dumps 1997 intermediates per forward pass.
- **DUMP_GQA_DEBUG_DIR** added to bytropix `wubu_gqa_forward()` — per-layer GQA intermediate dumps via `DUMP_GQA_LAYER` env var.
- **Per-layer comparison** shows divergence starts from L0 (cs=0.405), not L31. Both systems produce same output token for "Hello" ("," token 11).
- **gen_text symlink** created: `gen_text → gen_text_cpu`.

## Root Cause Analysis
Hidden states diverge from L0 (cs=0.405 across all 2048 dims of first layer output). This is NOT a GQA-specific issue. Likely:
1. Token embedding lookup differs between bytropix and llama.cpp
2. L0 SSM computation path differs
3. Quantization accuracy in early layers compounds through 30 SSM layers

The L31 attention probe (cs=0.471) is cleaner than its neighbors (L30 cs=0.182, L32 cs=0.504). L31 is NOT the primary failure point.

## Debug Tools Built
| Tool | Env Var | Where |
|------|---------|-------|
| Reference intermediate dumps | `DUMP_INTERMEDIATE_DIR` | llama.cpp (rebuilt) |
| Byropix GQA intermediates | `DUMP_GQA_DEBUG_DIR` + `DUMP_GQA_LAYER` | `src/wubu_ssm.c` |
| Byropix per-layer hidden | `DUMP_LAYER_DIR` | `src/wubu_model.c` (already existed) |

## Next Session Priority
1. **Compare token embeddings** between bytropix and llama.cpp — add `DUMP_EMBEDDING` to bytropix `gen_text_cpu` and compare against reference `global_model.input_embed.bin`.
2. **Trace L0 SSM** — if embeddings match, add SSM intermediate dumps to `wubu_ssm_forward()` to find SSM divergence source.
3. **Final output parity** — if embedding and L0 match, proceed to deeper layers.

## Key Build Commands
```
make gen_text_cpu                               # CPU binary
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N   # coherent path
DUMP_INTERMEDIATE_DIR=/tmp/ref ./llama-simple -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -n 1 "Hello"  # reference dump
DUMP_LAYER_DIR=/tmp/layers FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "Hello" 1  # bytropix layer dump
```

## Known sm_120 Bugs
1. `static __shared__` inside loops hangs — use `extern __shared__` with manual offsets.
2. `__syncthreads()` after warp-leader shared-write hangs — use serial reduction by thread 0.
3. `extern __shared__ uint8_t` with syncthreads in loops causes incorrect codegen — use `float*`.
