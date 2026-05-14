# WuBuText AI — Project Overview

## Purpose
Mission statement, done/pending list, constraints.

---

## Mission

Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry. Verify every layer against llama.cpp reference. Train with Token-Superposition Training (TST). Deliver inference training on consumer GPU (RTX 5050 6.4GB).

## Done ✅

| Phase | Component | Status | Evidence |
|-------|-----------|--------|----------|
| Phase 0 | GGUF reader | ✅ 13 types supported | train_real loads all 733 tensors |
| Phase 1 | Embedding graft (Poincaré) | ✅ R=0.956 | data/qwen36_embeddings_c.bin (1.9GB) |
| Phase 2 | SSM forward (30 layers) | ✅ CPU | train_real CE 12.66 |
| Phase 2 | GQA forward (10 layers) | ✅ CPU | train_real all 40 layers |
| Phase 3 | CE loss computation | ✅ streaming 248K vocab | train_real: 12.66 |
| Phase 3 | Tokenizer | ✅ CJK round-trip | test_tokenizer PASS |
| Phase 3 | MoE forward | ✅ CPU [-0.028, 0.031] | test_moe PASS |
| Phase 3 | MMProj dump | ✅ | dump_mmproj PASS |

## Broken ⛔

| Component | Failure | Root Cause | Priority |
|-----------|---------|------------|----------|
| GPU weight loading | bench_e2e all zeros | bench.c gpu_load_ssm_layer | P0 |
| GPU training | train_gpu CE 69 vs 12.66 | Same as above | P0 |
| Gradient training | train_backprop hangs at 180s | Unknown (code path identical) | P1 |

## Constraints

- **English only** — no CJK in code/comments/communication
- **Pure C + CUDA** — no Python for core logic
- **Verify ALL claims** — run binary, paste output, never "should work"
- **YOLO mode** — no questions, execute-verify-report loop
