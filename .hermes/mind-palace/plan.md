# WuBuText AI — Plan (May 15 PM v7)

## Purpose
All tests passing. Text inference pipeline built. 256K context: MoE router verified, SSM O(T), GQA needs KV cache. This plan covers the remaining bottlenecks for long-context inference and quality generation.

---

## Priority Queue

### P0 — KV Cache for GQA Inference
Current GQA recomputes full attention O(T²) per step. KV cache = O(T) memory, O(1) per-step compute.
- **Design:** Cache K/V tensors per GQA layer. On each step, append new K/V, run attention over cache.
- **Impact:** Enables autoregressive generation at long context. Without it, 256K is impractical.
- **Size:** 10 GQA layers × 2 KV heads × 256 head_dim × 256K context × 4 bytes = ~5GB
- **Risk:** Memory pressure on RTX 5050 (6.4GB total)

### P1 — Lazy Per-Expert MoE for Inference
Current `infer_text.c` with MoE does full per-layer dequant per step → 2560 load/free cycles for 64 tokens.
- **Design:** Cache top-k expert weights per layer between steps. Reuse if expert set doesn't change.
- **Impact:** Enables meaningful text generation (without MoE, FFN is passthrough → garbage output)
- **Reference:** `infer_moe_lazy.c` pattern — dequant only unique experts per token batch

### P1 — Move Output Projection to GPU
CPU output projection: O(V×D) = 248320 × 2048 = ~500M MACs per token. ~2s/token.
- **Design:** Use cuBLAS SGEMM on GPU for output projection
- **Impact:** 100×+ speedup (GPU does 500M MACs in ~0.5ms)

### P2 — PGA LR Tuning
PGA backward gradients extreme (dQ=1.95, dK=0.004, dV=0.70, dX=571). Current lr_gqa=lr*0.01=1e-5 too high.
- **Fix:** Try lr_gqa = lr * 0.001 or gradient clipping at norm=1.0
- **Impact:** Steps 2+ would not jump from CE 21.6→69 (currently stuck)

### P2 — Multi-Step Convergence (50+ steps)
Current verification only at 2-3 steps. Need to verify:
- No long-term NaN emergence
- CE steadily decreasing (target < 5.0 after many steps)
- Embedding norms stable (no drift to Poincaré boundary)

### P3 — MRoPE (Multi-Resolution RoPE) Implementation
Qwen3.6 uses `mrope_interleaved=true`, `mrope_section=[11,11,10]` (32 total 3D positional dims).
Standard RoPE with partial_rotary_factor=0.25 works for text. MRoPE needed for vision-text interleaving.
- **Impact:** Better position encoding for long context (>32K) and multi-modal

### P3 — Sparse Attention Port (O(n·k) linear, highest ROI vault port)
PyTorch prototype in vault. Could replace GQA for long context.

---

## 256K Context Roadmap

| Step | What | Depends On |
|------|------|------------|
| 1 | GQA KV cache (append-only) | — |
| 2 | SSM state carry (already works) | — |
| 3 | Verify 256K forward pass (no generation) | KV cache |
| 4 | Single-token generation at 256K | Step 3 |
| 5 | Lazy MoE for inference | — |
| 6 | GPU output projection | — |
| 7 | Coherence test with MoE @ 256K | Steps 4-6 |

## Known Blockers

| Blocker | Reason |
|---------|--------|
| KV cache ~5GB | RTX 5050 has 6.4GB total. Model weights already use ~1.2GB GPU. Tight fit. |
| Without MoE, output is garbage | MoE is the only FFN. Attention/SSM alone can't produce meaningful text. |
| MoE inference is 40× slow | Full dequant per layer × 40 layers × 64 steps = 2560 load/free cycles |
