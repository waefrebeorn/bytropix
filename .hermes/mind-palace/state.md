# State — May 21 PM (Phase 29c: DUMP_INTERMEDIATE_DIR Hooks + Per-Layer Divergence Audit)

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

## Session Progress — Phase 29c

**Accomplished:**
1. **Llama.cpp rebuilt** with DUMP_INTERMEDIATE_DIR support — `libllama.so` + `llama-simple` (fast, no chat template). Verified 1997-intermediate tensor output per forward pass.
2. **DUMP_GQA_DEBUG_DIR** added to `wubu_gqa_forward()` — dumps input, Q_full, gate, K, V, Q_norm, K_norm, attn_out (pre+post gate), output for each GQA layer. Supports `DUMP_GQA_PREFIX` for per-layer filenames.
3. **DUMP_GQA_LAYER** env var in `wubu_model.c` — sets `DUMP_GQA_PREFIX` only for target layer (e.g., `DUMP_GQA_LAYER=31` → files prefixed `L31_gqa7_*`). Other layers get empty prefix (suppressed).
4. **gen_text symlink** created: `gen_text → gen_text_cpu`
5. **Per-layer hidden state comparison** shows divergence starts from **L0** (cs=0.405). L0-L30 cs drifts to ~0.17. L31 improves to cs=0.47. Both produce same output token ("," = token 11 for "Hello").

## What's Broken
- **Chunked SSM CS>1**: FP accumulation across 30 SSM layers → wrong tokens. Only CS=1 is exact.
- **GPU text net-negative**: H2D/D2H overhead + thermal throttling makes GPU hybrid 2-5x slower than CPU.
- **1:1 parity unresolved**: Hidden states diverge from L0 (cs=0.405). Likely root cause: token embedding or first SSM layer. L31 attention is NOT the primary culprit.

## Key Env Vars
```
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N   # sequential SSM (coherent)
ROPE_SCALE_FACTOR=0.25                           # 4x context extension
USE_SPARSE_ATTN=1 SPARSE_W=512 SPARSE_G=128      # NSA sparse attention
DUMP_GQA_DEBUG_DIR=/tmp/gqa DUMP_GQA_LAYER=31    # debug L31 GQA intermediates
DUMP_LAYER_DIR=/tmp/layers                        # dump per-layer hidden states
DUMP_INTERMEDIATE_DIR=/tmp/ref                    # llama.cpp reference dump
```

## Debug Infrastructure Built This Session
- `DUMP_INTERMEDIATE_DIR` in llama.cpp (works, but buffer reuse may corrupt some tensors — use `l_out` for reliable per-layer comparison)
- `DUMP_GQA_DEBUG_DIR` + `DUMP_GQA_PREFIX` + `DUMP_GQA_LAYER` in bytropix (`src/wubu_ssm.c`, `src/wubu_model.c`)
- `DUMP_LAYER_DIR` already existed in bytropix (per-layer hidden states)

## Per-Layer Cos-Sim (vs llama-simple, "Hello" 1-token)
L0: 0.405 | L1: 0.445 | L2: 0.664 | L3: 0.568 | L6: 0.445 | L10: 0.142 | L15: 0.175
L20: 0.197 | L25: 0.188 | L30: 0.182 | L31: 0.471 | L35: 0.362 | L39: 0.496

Divergence starts at L0 → root cause is NOT L31 GQA but token embedding or first SSM layer.

## COMMITS (latest)
- ec58b72 — docs: Phase 29a state — IQ1_M + Q4_K GPU kernels
- c0254c0 — feat(gpu): IQ1_M + Q4_K quant matmul kernels, CPU IQ1_M fallback
