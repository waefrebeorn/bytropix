# 4. Implementation Status

**Date:** May 13, 2026 — DA Audit Update
**Tone:** Conservative — verified claims, DA-caught stale claims identified.

---

## Phase 0: GGUF Reader ✅

**Files:** `src/gguf_reader.c`, `include/gguf_reader.h`

**What works:**
- Full GGUF format parsing (13 GGML types: F32, Q4_0–Q8_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S)
- Tensor extraction by name — verified against Qwen3.6-35B-A3B GGUF (733 tensors)
- Q4_K/Q5_K dequant fixed: `block_q4_K` has NO qh field (144 bytes, not 176). This was the ROOT CAUSE of all NaN and garbage output in prior sessions.
- All 733 tensors of 11GB model load and dequant correctly on CPU.

**DA Caught:** Old claims about "IQ2 dequant garbage" were wrong — Q4_K was the root bug.

---

## Phase 1: Embedding Graft ✅

**Files:** `src/wubu_mobius.c`, `include/wubu_mobius.h`, `data/qwen36_embeddings_c.bin`

**What works:**
- Euclidean → Poincaré exponential mapping at radius R=0.956
- ~95% nearest-neighbor preservation
- Embeddings file on disk: 1.9GB, 248K tokens, ready for training

---

## Phase 2: SSM/GQA Forward Pass (CPU) ✅ — (GPU ⛔)

**Files:** `src/wubu_ssm.c`, `src/cuda_kernels.cu`, `src/bench.c`, `src/wubu_model.c`

**CPU forward (works ✅):**
- All 40 layers (30 SSM + 10 GQA) forward on CPU via `wubu_model_forward_from_embd`
- CE loss: 12.66 (near random baseline 12.42) — logits non-zero
- 0.2 tok/s CPU throughput

**GPU forward (broken ⛔):**
- `bench_e2e` produces ALL ZEROS for both CPU and GPU paths (max val 0.000000)
- **Root cause:** GPU weight loading functions (`gpu_load_ssm_layer`, `gpu_load_gqa_layer` in bench.c) read wrong data from GGUF
- `train_gpu` produces CE loss 69 vs expected 12.4 — same root cause

**DA Caught:** Prior docs claimed "GPU/CPU agreement verified at 9.53 tok/s" — this was a false positive from zero-output comparison.

---

## Phase 3: Training Loop 🔄

**Files:** `tools/train_real.c`, `tools/train_backprop.c`, `tools/train_gpu.c`

**What works:**
- Tokenizer implemented: GPT-2 BPE, CJK round-trip, 248K vocab, merge hash (11% collision rate)
- train_real: CPU forward + CE loss + output projection (508M elements). CE loss 12.66.
- test_moe: 256 experts + shared expert, output [-0.028, 0.031], NaN=0, 36.6 tok/s CPU.

**What's broken:**
- train_backprop: hangs during model init (≥180s timeout)
- train_gpu: GPU forward produces CE 69 (wrong)
- MoE not integrated into training loop

---

## Phase 2.5: GPU Verification ⚠️

**Status:** Revoked. Prior "verified" claims were based on all-zero output from broken GPU weight loading. CPU forward is verified (train_real). GPU verification requires P0 fix.

---

## Key DA Audit Findings (May 13)

1. **5 of 8 binaries PASS** with fresh verified output
2. **3 binaries fail** — all traceable to GPU weight loading bug or train_backprop hang
3. **Old claims debunked:** "CE loss commented out" ❌, "IQ2 garbage" ❌, "bench_e2e 29x PASS" ❌
4. **Real P0:** Fix GPU weight loading in bench.c
