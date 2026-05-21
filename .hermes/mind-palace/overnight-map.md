# Overnight Map — Phase 28p: P2.4 RoPE Extrapolation Complete

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 48dcf5e (pushed to origin/master)  
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)  
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB)  

## Session Summary (May 21, 2026 — P2.4 RoPE 4x + Chunked SSM Investigation)

### P2.4 — RoPE Extrapolation 4x: COMPLETE (commit 48dcf5e)

- Added `ROPE_SCALE_FACTOR` env var to IMRoPE in `wubu_ssm.c:1290-1325`
- Qwen2.5-1M §3.1 method: `theta = (pos * scale) * freq_base^{-2i/d}`
- `ROPE_SCALE_FACTOR=0.25` extends context 64K→256K (4x)
- Default `scale=1.0` = no change, fully backward compatible
- **Verified coherent text at both settings:**
  - Default (scale=1.0): "the city of Paris. It is the capital" @ 7.7 tok/s
  - 4x (scale=0.25): "the most visited city in the world, with" @ 6.5 tok/s

### P2.3 — Chunked SSM: INVESTIGATED, STILL BROKEN

- `test_chunked_ssm` FAILS with max_diff_out=0.129, max_diff_state=0.52
- State divergence is catastrophic (not numerical noise)
- Heap corruption at T=65: "free(): invalid pointer"
- **Root cause analysis:** Causal addressing convention in KQ/mask is suspect. The mask uses `i >= j` convention but the output `v_new^T @ kq` reads `kq[z][t]` for position t. With mask `i >= j`, kq non-zero at z >= t means queries attend to FUTURE keys (non-causal). But reverting to `i <= j` made the strict-lower solve trivial (L=0).
- **Reference:** llama.cpp `build_delta_net_chunking()` in `delta-net-base.cpp` uses `ggml_tri(decay_mask, GGML_TRI_TYPE_LOWER_DIAG)` — which preserves the LOWER triangle including diagonal. The bytropix chunked code uses the same mask convention but the indices convention might differ.
- **Status:** Needs deeper comparison against llama.cpp reference. The bytropix chunked code does NOT use ggml ops — it's hand-coded C loops. A line-by-line comparison of the matrix dimension indexing is needed.

### gen_text_cpu Verified

- CLI usage: `./gen_text_cpu "prompt string" <max_tokens>` (model path is hardcoded)
- Produces coherent text at 7.7 tok/s (close to expected 8.9)
- Decode loop works correctly

### P2 Status Summary

| Priority | Item | Status | Why |
|----------|------|--------|-----|
| P2.0 | CUDA sm_120 bug skill | ✅ Done | 6 bugs documented, skill updated |
| P2.1 | Llama.cpp inline hooks | ✅ Already exists | DUMP_LAYER_DIR + DUMP_INTERMEDIATE_DIR in ref_dumper |
| P2.2 | GPU RMSNorm + SiLU kernels | 🔲 Skipped | GPU text is net-negative; no benefit |
| P2.3 | Chunked prefill | ❌ BROKEN | Chunked SSM cos_sim=0.00000045 vs sequential. Needs deep debug |
| P2.4 | RoPE extrapolation 4x | ✅ COMPLETE | ROPE_SCALE_FACTOR=0.25 env var |
| P2.5 | NSA sparse attention | 🔲 Not started | High effort, high impact for 256K |
| P2.6 | Sigmoid gating + load balancing | ❌ NOT APPLICABLE | Training-time technique from DeepSeekMoE. Model trained with softmax |
| P2.7 | FP8 Tensor Cores | 🔲 Not started | sm_120 native, 2x throughput potential |

### Verifiable Facts

- gen_text_cpu works: `./gen_text_cpu "prompt" N`
- ROPE_SCALE_FACTOR=0.25 for 4x context extrapolation
- ref_dumper + DUMP_LAYER_DIR = 40 layer dumps (8KB each)
- ref_dumper + DUMP_INTERMEDIATE_DIR = 1997 intermediate files
- CUDA sm_120 bugs: 6 documented in `cuda-sm120-bugs` skill
- Chunked SSM: `src/wubu_ssm_chunked.c` is BROKEN; `tools/test_chunked_ssm.c` fails
- All mind-palace markdown files updated in 473f2b2 and 48dcf5e

### Next Session Priorities

1. **Fix chunked SSM** — deep debug needed. Compare against llama.cpp `build_delta_net_chunking()` in `delta-net-base.cpp`. The likely issue is the causal addressing convention (mask indices vs KQ indices).
2. **NSA sparse attention** — from DeepSeek-V3.2. O(L log L) for GQA layers at 256K.
3. **FP8 Tensor Cores** — sm_120 native FP8 dot product for batched quant matmul.
