# Overnight Map — Phase 28r: P2.3 Chunked SSM Wired + Limited

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 501518f (pushed to origin/master)  
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)  
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB)  

## Session Summary (May 21, 2026 — P2.3 Chunked SSM Fixed + Wired)

### What Was Done
1. **Data layout bug FIXED** — memcpy assumed head-contiguous layout, but caller stores token-interleaved. Token-by-token extraction fix.
2. **Cyclic repeat mapping FIXED** — changed from sequential (`vh / rf`) to cyclic (`vh % hk`) to match inline forward.
3. **cur_nt bounds FIXED** — used `nt` (padded) not `T` (real), causing OOB write on last chunk.
4. **Chunked SSM wired into `wubu_ssm_forward()`** — T >= CS triggers chunked path. `SSM_CHUNK_MIN` env var. `FORCE_CPU_SSM_SEQ` to disable.
5. **Verified CS=1 exact** — test passes with 4e-8 max diff.
6. **CS=2/8/64 tested with real model** — all produce wrong tokens. FP accumulation across 30 SSM layers × many ops amplifies beyond usable threshold.

### P2 Status Summary

| Priority | Item | Status | Why |
|----------|------|--------|-----|
| P2.0 | CUDA sm_120 bug skill | ✅ Done | 6 bugs documented |
| P2.1 | Llama.cpp inline hooks | 🔲 Not started | 1 session |
| P2.2 | GPU RMSNorm + SiLU kernels | 🔲 Skipped | GPU text net-negative |
| P2.3 | Chunked prefill | ✅ Wired + limited | CS=1 exact only. CS>1 FP-limited |
| P2.4 | RoPE extrapolation 4x | ✅ Complete | `ROPE_SCALE_FACTOR=0.25` |
| P2.5 | NSA sparse attention | 🔲 Not started | High effort for long context |
| P2.6 | Sigmoid gating | ❌ N/A | Training-time only |
| P2.7 | FP8 Tensor Cores | 🔲 Not started | Needs F >> D overhead solved first |

### Verifiable Facts
- `FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N` — exact sequential path
- `./gen_text_cpu "prompt" N` — chunked for T >= 2 (default CS=2)
- `SSM_CHUNK_MIN=64 ./gen_text_cpu "long prompt" N` — override min threshold
- Chunked SSM skill saved: `chunked-ssm-debug-pattern`

### Next Session Priorities
1. **NSA sparse attention** — O(L·(w+g)) for GQA layers. DSA pattern from DeepSeek-V3.2.
2. **FP8 Tensor Cores** — sm_120 native. Low impact until GPU text data-movement problem solved.
3. **Llama.cpp inline hooks** for richer reference data.
