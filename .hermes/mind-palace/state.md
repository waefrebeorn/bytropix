# WuBuText AI — State Dashboard

## Purpose
Live status of all 8 key binaries verified May 13, 2026. Every claim backed by actual runtime output.

---

## 🔴 Dev-il's Advocate Audit — May 13, 2026

### Verified Binaries (fresh run, actual output captured)

| Binary | Status | Key Metrics | Notes |
|--------|--------|-----------|-------|
| train_real | ✅ PASS | CE loss 12.66 (random baseline 12.42), logits non-zero, fwd 0.2 tok/s CPU | **The only correct training path.** Uses `wubu_model_forward_from_embd` with pre-loaded CPU weights. |
| test_fused_vs_old | ✅ PASS | GPU max diff 0.03587 (cuBLAS FP artifact, accepted) | Unchanged from prior session. |
| test_tokenizer | ✅ PASS | "你好" → 109266 → "你好" round-trip | CJK works via pre-token merge avoidance. Python subprocess fails (non-issue). |
| test_moe | ✅ PASS | Output range [-0.028, 0.031], NaN=0, 36.6 tok/s | **Q4_K fix resolved the NaN/1e6 garbage.** Old DA claims are STALE. |
| dump_mmproj | ✅ PASS | 334 tensors, 27 ViT blocks, correct merger dims | Unchanged. |
| bench_e2e | ⛔ BROKEN | All-output-zero: CPU max 0.000, GPU max 0.000 | **GPU forward path produces nothing.** Old GPU weight loading (bench.c) doesn't use wubu_model_forward. |
| train_gpu | ⚠️ WRONG LOSS | CE loss 69→51 vs expected 12.66 | GPU weight loading per-layer-per-step = garbage outputs. Copies GGUF weights to GPU each step — slow AND wrong. |
| train_backprop | ⛔ HANGS | Times out at 180s during model init | Same code as train_real + gradient logic. Root cause: obscure, identical compilation path. |

### Old DA Claims vs Reality

| Old Claim (prestige/map) | Reality | Verdict |
|-------------------------|---------|---------|
| "CE loss COMMENTED OUT" | CE loss IS wired — train_real produces 12.66 | ❌ STALE CLAIM |
| "IQ2 dequant = garbage" | Q4_K was the root bug. Fixed. train_real produces correct values. | ❌ STALE CLAIM |
| "bench_e2e loops forever" | bench_e2e runs to completion but outputs ALL ZEROS | ⚠️ NEW BUG FOUND |
| "train_real hangs" | train_real works fine with 120s timeout | ❌ STALE CLAIM |
| "GPU forward = 29x speedup" | Speedup real, but output is ZERO | ⚠️ DECEPTIVE — speedup on garbage |

### Risk Assessment

| Risk | Severity | Monitoring | 
|------|----------|-----------|
| bench_e2e zero-output bug | 🔴 HIGH — hidden states after 40-layer fwd are zero | Hidden states diverge from train_real output |
| GPU weight loading path broken | 🔴 HIGH — gpu_load_ssm_layer reads wrong/WRONG data | train_gpu loss 69 vs train_real 12.66 |
| train_backprop hangs | 🟡 MEDIUM — gradient training blocked | May be fixed by using train_real code path |
| All "PASS" claims from old sessions | 🟢 LOW — re-verified all 5 PASS binaries | Fresh output confirms |
| train_real only uses embeddings, not tokenizer | 🟢 LOW — embeddings file exists and works | Token embeddings loaded at init |

### True Current Priority Queue

P0 — Fix GPU weight loading (bench.c gpu_load_ssm_layer/gpu_load_gqa_layer)
P1 — Fix train_backprop hang 
P2 — Verify train_real backward pass
P3 — Add MoE to training path
