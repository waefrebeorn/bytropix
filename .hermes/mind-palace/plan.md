# WuBuText AI — Plan (May 14 PM)

## Purpose
Current priorities based on inference phase completion.

---

## P0 — Fix GPU weight loading (bench.c → zeros)

**Symptom:** bench_e2e all zeros, train_gpu CE 69 vs 12.66.
**Root cause:** gpu_load_ssm_layer/gpu_load_gqa_layer read wrong data from GGUF.

**Fix:** Replace bench.c weight loading with wubu_model_init path (which works correctly).

---

## P1 — Fix training pipeline

**After P0:** rebuild train_gpu with fixed GPU forward.
**TGT wrapping already applied to SGD step** (replaced clip[-10,10] with π-odometer).

**Verify:** CE loss ≈ 12.4-12.7 (matches train_real).

---

## P2 — Fix train_backprop hang

**Symptom:** Times out at 180s during model init.
**Root cause:** Unknown. Same source as train_real but hangs.

---

## P3 — Vision→model integration

**Status:** mmproj extracted, GPU ViT working (217ms).
**Need:** Wire mmproj output into 40-layer model pipeline.

---

## P4 — Integrate lazy MoE into training

**Status:** infer_moe_lazy works for inference. Need training equivalent.
**Benefit:** Eliminates 120-expert dequant bottleneck (9× speedup).
