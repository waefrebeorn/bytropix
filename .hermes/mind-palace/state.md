# State — May 21 PM (Phase 29d: Token Embedding Divergence — Root Cause Found)

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**  
**Reference: llama.cpp (libllama.so, DUMP_INTERMEDIATE_DIR in llama-simple)**  
**CUDA: sm_120 (RTX 5050 Blackwell, 13.1 toolkit)**  
**Only model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB)**

## CURRENT STATE

| Component | Result | Status |
|-----------|--------|--------|
| CPU text (FORCE_CPU_SSM_SEQ=1) | "the city of Paris..." | ✅ Verified coherent, 3-4 tok/s |
| CPU text (default — chunked SSM) | "the law, the 1949..." | ❌ CS>1 FP accumulation breaks output |
| GPU vision encoder (ViT + MMProj) | GPU ViT 0.52s, total 15.7s | ✅ GPU accelerated 4.4x |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ Working (quantized head) |
| GPU SSM/GQA + CPU MoE hybrid | Coherent text at 5.5 tok/s | ✅ Working |
| GPU quant matmul (Q5_K/Q6_K/Q4_K) | 3 types, single+batched | ✅ Kernels exist |
| gen_text_gpu build | Links without errors | ✅ Builds |

## Session Progress — Phase 29d: Token Embedding Root Cause

**Accomplished:**
1. **DUMP_EMBEDDING_DIR** added to `tools/gen_text.c` — dumps embedding buffer right after token lookup.
2. **Token embedding comparison** reveals cs=0.118 between bytropix and llama.cpp:
   - Reference `global_model.input_embed.bin`: mean=0.011, std=1.295 (correct F32 dequant)
   - Bytropix `embedding.bin`: mean=3.1e-5, std=0.013 (nearly zero — quantization bug)
   - gguf_read_tensor_f32() fails to dequantize quantized token_embd.weight

## What's Broken
- **Token embedding**: gguf_read_tensor_f32 produces wrong values for quantized token_embd.weight (cs=0.118 vs reference). This is the ROOT CAUSE of the L0 cos-sim=0.405 divergence.
- **Chunked SSM CS>1**: FP accumulation across 30 SSM layers → wrong tokens. Only CS=1 is exact.
- **GPU text net-negative**: H2D/D2H overhead + thermal throttling makes GPU hybrid 2-5x slower than CPU.

## Key Env Vars
```
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N   # sequential SSM (coherent)
DUMP_EMBEDDING_DIR=/tmp/emb                       # dump token embedding
DUMP_GQA_DEBUG_DIR=/tmp/gqa DUMP_GQA_LAYER=31    # debug L31 GQA intermediates
DUMP_LAYER_DIR=/tmp/layers                        # dump per-layer hidden states
DUMP_INTERMEDIATE_DIR=/tmp/ref                    # llama.cpp reference dump
```

## Debug Infrastructure Built
- `DUMP_INTERMEDIATE_DIR` in llama.cpp (rebuilt libllama.so + llama-simple)
- `DUMP_GQA_DEBUG_DIR` + `DUMP_GQA_PREFIX` + `DUMP_GQA_LAYER` in bytropix (`src/wubu_ssm.c`, `src/wubu_model.c`)
- `DUMP_EMBEDDING_DIR` in bytropix (`tools/gen_text.c`)
- `DUMP_LAYER_DIR` in bytropix (per-layer hidden states)

## COMMITS (latest)
- 4ebe712 — feat(debug): DUMP_GQA_DEBUG_DIR + per-layer divergence audit
- ec58b72 — docs: Phase 29a state — IQ1_M + Q4_K GPU kernels
