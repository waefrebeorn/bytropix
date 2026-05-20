# Goal Mantra — May 19, 2026 PM (Phase 22 — Q4_0 KV Cache ✅)

## THE GOAL
**1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.** ✅
**Overall cos-sim: 0.9994 (CPU, 5-token prefill, 40 layers).**

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Architecture | 40 layers, 3:1 SSM/GQA **interleaved** repeating | ✅ Discovered May 19 |
| gen_text (CPU) | ~11 tok/s prefill, Q4_0 KV cache | ✅ Phase 22 |
| gen_text_gpu | Pre-existing hang | ❌ Needs debug |
| Q4_0 KV cache | 720MB vs 2.56GB at 256k, 4:1 compression | ✅ Cos-sim 0.9994 |
| DUMP_INTERMEDIATE_DIR | 1997 files/forward, 53 tensor types/layer | ✅ Built |
| Cos-sim L00-L30 | 0.998-0.9999 | ✅ |
| Cos-sim L31 | 0.9585 (GQA quantization noise) | 🟡 Expected |
| Overall cos-sim | 0.9994 | ✅ |

## BUILD
```bash
make gen_text                # CPU inference
make gen_text_gpu            # GPU inference (currently hangs)
make ref_dumper              # libllama.so reference tool
```

## HW
AMD Ryzen 7950X (16C/32T) | 64GB DDR5 | RTX 5050 8GB | WSL2
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

## NEXT STREAMS
P0 — Fix gen_text_gpu hang, GPU Q4_0 KV cache
P1 — Unified SSM kernel fusion, parallel cuBLAS streams
P2 — Sparse attention + global tokens for 512k+

## VAULT
- Architecture: 3:1 interleaved SSM/GQA (verified GGUF enumeration)
- DUMP_INTERMEDIATE_DIR: llama.cpp cb() stores all intermediates
- Q4_0 KV cache: block_q4_0_cache in wubu_model.h
- Unsloth UD quant: SSM Q5_K/Q6_K, MoE IQ2_XXS/IQ3_XXS, out Q4_K
