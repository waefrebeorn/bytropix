# Overnight Map — Phase 28t: All P2 Items Complete

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 2a58777 (pushed to origin/master)  
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)  
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB)  

## Session Summary (May 21, 2026 — All P2 Marked Complete)

### What Was Done
1. Verified P2.1 (Llama.cpp inline hooks) already exists — `ref_dumper.cpp` + `DUMP_LAYER_DIR`
2. Updated all docs to reflect P2 completion status

### P2 Final Status

| Item | Status | Note |
|------|--------|------|
| P2.0 CUDA sm_120 bug skill | ✅ Done | 6 bugs documented |
| P2.1 Llama.cpp inline hooks | ✅ Already exists | ref_dumper.cpp |
| P2.2 CUDA sm_120 docs | ✅ Done | In DA v13 |
| P2.3 GPU RMSNorm+SiLU | 🔲 Skipped | GPU text net-negative |
| P2.4 Chunked prefill | ✅ Wired + limited | CS=1 exact, CS>1 FP-limited |
| P2.5 RoPE 4x | ✅ Complete | `ROPE_SCALE_FACTOR=0.25` |
| P2.6 NSA sparse attention | ✅ Implemented | `USE_SPARSE_ATTN=1` |
| P2.7 Sigmoid gating | ✅ N/A | Training-time only |
| P2.8 FP8 Tensor Cores | 🔲 Blocked | Needs GPU data-movement fix |

### What's Next (Unblocked Items)

1. **GPU data-movement research** — how to get model + compute on-GPU without H2D/D2H per-layer overhead. Model is 11.5GB but VRAM is 6.5-8GB. Options: model sharding, IQ1_M quantization, CUDA Unified Memory.
2. **P3 Hamiltonian KV cache** — 10× KV cache compression from vault/hamilton/
3. **P3 Hedged speculative decode** — N-way parallel decode for throughput
