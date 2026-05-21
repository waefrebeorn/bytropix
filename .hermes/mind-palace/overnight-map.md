# Overnight Map — Phase 28s: NSA Sparse Attention Implemented

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 0129f1a (pushed to origin/master)  
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)  
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB)  

## Session Summary (May 21, 2026 — NSA Sparse Attention Implemented)

### What Was Done
1. **NSA sparse attention for GQA layers** — DeepSeek-V3.2 DSA pattern:
   - S(i) = local_window(i, W) ∪ global_positions(i, G)
   - Configured via env vars: `USE_SPARSE_ATTN`, `SPARSE_W`, `SPARSE_G`, `SPARSE_MIN`
   - Verified coherent text: "Today, AI systems can generate text, images, and even code"
   - Reduces O(L²) to O(L·(W+G)) — critical at 256K context

### P2 Status Summary

| Priority | Item | Status | Why |
|----------|------|--------|-----|
| P2.0 | CUDA sm_120 bug skill | ✅ Done | 6 bugs documented |
| P2.1 | Llama.cpp inline hooks | 🔲 Not started | 1 session |
| P2.2 | GPU RMSNorm + SiLU kernels | 🔲 Skipped | GPU text net-negative |
| P2.3 | Chunked prefill | ✅ Wired + limited | CS=1 exact only |
| P2.4 | RoPE extrapolation 4x | ✅ Complete | `ROPE_SCALE_FACTOR=0.25` |
| P2.5 | **NSA sparse attention** | ✅ **Implemented** | 0129f1a |
| P2.6 | Sigmoid gating | ❌ N/A | Training-time only |
| P2.7 | FP8 Tensor Cores | 🔲 Not started | Needs GPU data-movement solved |

### Next Session Priorities
1. **Test sparse attention at 4K+ context** — verify it handles the full GQA load
2. **Llama.cpp inline hooks** — richer reference data for SSM verification
3. **FP8 Tensor Cores** — or pick from P3 (Hamilton KV cache)
