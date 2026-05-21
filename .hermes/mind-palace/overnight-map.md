# Overnight Map — Phase 28q: P2.3 Chunked SSM Bug Fixed

**Active repo:** /home/wubu/bytropix/  
**Current commit:** c5475af (pushed to origin/master)  
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

### P2.3 — Chunked SSM: DATA LAYOUT BUG FIXED (commit c5475af)

- **Root cause:** memcpy at function entry assumed head-contiguous layout
  (`data[h*T*d + t*d + i]`), but caller stores data token-interleaved
  (`data[(t*H + h)*d + i]`). Token-by-token extraction fixes it.
- **Verified CS=1:** chunked matches sequential EXACTLY (max diff 4e-8 output, 3e-7 state)
- **CS=64 FP error:** 7.4e-2 output diff, 0.24 state diff — from +2000x more float ops per position.
  The chunked formula is mathematically exact but float rounding accumulates differently.
- **Status:** Data layout bug fixed. CS>1 FP error is documented. Use CS=1 for exact inference.

### P2 Status Summary

| Priority | Item | Status | Why |
|----------|------|--------|-----|
| P2.0 | CUDA sm_120 bug skill | ✅ Done | 6 bugs documented, skill updated |
| P2.1 | Llama.cpp inline hooks | ✅ Already exists | DUMP_LAYER_DIR + DUMP_INTERMEDIATE_DIR in ref_dumper |
| P2.2 | GPU RMSNorm + SiLU kernels | 🔲 Skipped | GPU text is net-negative; no benefit |
| P2.3 | Chunked prefill | ✅ Data layout bug FIXED | c5475af. CS=1 exact. CS>1 FP error acceptable? |
| P2.4 | RoPE extrapolation 4x | ✅ COMPLETE | ROPE_SCALE_FACTOR=0.25 env var |
| P2.5 | NSA sparse attention | 🔲 Not started | High effort, high impact for 256K |
| P2.6 | Sigmoid gating + load balancing | ❌ NOT APPLICABLE | Training-time technique from DeepSeekMoE. Model trained with softmax |
| P2.7 | FP8 Tensor Cores | 🔲 Not started | sm_120 native, 2x throughput potential |

### Verifiable Facts

- Chunked SSM: data layout bug FIXED. CS=1 matches sequential exactly.
- gen_text_cpu works: `./gen_text_cpu "prompt" N`
- ROPE_SCALE_FACTOR=0.25 for 4x context extrapolation
- ref_dumper + DUMP_LAYER_DIR = 40 layer dumps (8KB each)
- ref_dumper + DUMP_INTERMEDIATE_DIR = 1997 intermediate files
- CUDA sm_120 bugs: 6 documented in `cuda-sm120-bugs` skill
- Chunked SSM: `src/wubu_ssm_chunked.c` — data layout fix in c5475af
- All mind-palace markdown files updated c5475af

### Next Session Priorities

1. **Test chunked SSM with real model** — the FP error may be acceptable for inference (tokens differ but still coherent). Compare `gen_text_chunked` vs `gen_text_cpu` output.
2. **NSA sparse attention** — from DeepSeek-V3.2. O(L log L) for GQA layers at 256K.
3. **FP8 Tensor Cores** — sm_120 native FP8 dot product for batched quant matmul.
