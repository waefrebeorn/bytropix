# WuBuText AI — Plan (May 16 AM v9)

## Purpose
All tests passing. `infer_text` v2 with GQA KV cache + lazy MoE + SSM state carry. 256K context feasible via KV cache (O(T) decode). Remaining: GPU forward acceleration, tailslayer speculative decode, PGA LR tuning.

---

## Priority Queue

### P1 — GPU Forward Acceleration for Decode
Current decode is CPU-only: 40 layers × (GQA/SSM + MoE) per token. GPU kernels already exist (`gpu_gqa_forward`, `gpu_ssm_forward` in bench.h/c).
- Wire GPU GQA forward with KV cache (GPU K/V buffers already in place)
- Wire GPU SSM forward with state carry (states already in GPU memory for training)
- **Impact:** 10-100× speedup (GPU does 40 layers in <10ms vs 7s CPU)

### P1 — Tailslayer Speculative Decode
N parallel draft tokens → longest-valid-prefix verification → forward-pass integration.
- N drafts from same model (no separate draft model needed — run N tokens through single forward pass)
- Longest-valid-prefix: verify all N drafts, take longest matching prefix
- clflush (CPU cache flush verification) → integrate into GPU forward pass
- Sliding window pair sampling for training data
- tREFI probe for CUDA DRAM profiling

### P2 — PGA LR Tuning
PGA backward gradients extreme (dQ=1.95, dK=0.004, dV=0.70, dX=571). Current lr_gqa=lr*0.01=1e-5 too high.
- **Fix:** Try lr_gqa = lr * 0.001 or gradient clipping at norm=1.0
- **Impact:** Steps 2+ would not jump from CE 21.6→69 (currently stuck)

### P2 — Multi-Step Convergence (100+ steps)
Current verification only at 2-50 steps. Need to verify:
- No long-term NaN emergence
- CE steadily decreasing (target < 5.0 after many steps)
- Embedding norms stable (no drift to Poincaré boundary)

### P3 — MRoPE (Multi-Resolution RoPE)
Qwen3.6 uses mrope_interleaved=true, mrope_section=[11,11,10]. Standard RoPE with partial_rotary_factor=0.25 works for text.

## 256K Context Roadmap

| Step | What | Depends On |
|------|------|------------|
| 1 | GQA KV cache (append-only) | ✅ Done |
| 2 | SSM state carry | ✅ Done |
| 3 | Lazy MoE cache (no 3GB arrays) | ✅ Done |
| 4 | GPU forward for GQA/SSM decode | GPU kernels exist, need wiring |
| 5 | Verify 256K forward pass (no generation) | Step 4 |
| 6 | Single-token generation at 256K | GPU forward |
| 7 | Tailslayer spec decode (N× speedup) | Step 4 |
| 8 | Coherence test with MoE @ 256K | Steps 4-7 |

## Known Blockers

| Blocker | Reason |
|---------|--------|
| No GPU forward in decode | All 40 layers run on CPU → 7s/token |
| KV cache ~5GB at 256K | RTX 5050 6.4GB total. Model weights ~1.2GB GPU. Tight fit. |
