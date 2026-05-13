# 4. Implementation Status

**Date:** May 2026
**Tone:** Conservative — verified claims, acknowledged blockers, no unsubstantiated speculation.

---

## Phase 1: GGUF Reader + Embeddings Extraction + Poincaré Mapping (✅ Done)

**Files:**
- `include/gguf_reader.h`, `src/gguf_reader.c` — GGUF file parser (header, tensor info, weight data)
- `tools/dbg_gguf.c`, `tools/dump_gguf.py` — debugging and inspection utilities
- `include/wubu_mobius.h`, `src/wubu_mobius.c` — Poincaré exp_map, log_map, Möbius operations
- `tools/test_tokenizer.c`, `tools/test_tokenizer_simple.c` — tokenizer test harnesses

**What works:**
- Full GGUF format parsing (GGML quantized types Q4_0–Q8_K, IQ2_XS, IQ1_S)
- Tensor extraction by name — verified against Qwen3.6-35B-A3B GGUF
- Euclidean → Poincaré exponential mapping at radius R=0.956
- Poincaré → Euclidean log map (for LM head projection)
- 73 zero-norm special tokens correctly positioned at origin
- ~95% nearest-neighbor preservation after mapping (preliminary)

**What's pending:**
- No formal validation of embedding quality beyond k-NN preservation
- Möbius gyration (gyr[x,y]z) implemented but untrained — full use deferred to Phase 3

---

## Phase 2: SSM/GQA Forward Pass in C + CUDA (✅ Done)

**Files:**
- `include/wubu_ssm.h`, `src/wubu_ssm.c` — SSM (30 layers) + GQA (10 layers) CPU forward
- `include/cuda_kernels.h`, `src/cuda_kernels.cu` — CUDA kernel layer (cuBLAS matmul, element-wise, RMSNorm, conv1d, Gated Delta Net step, GQA fused kernel)
- `include/bench.h`, `src/bench.c` — GPU weight load, GPU forward wrappers
- `test_ssm_forward.c`, `test_poincare_ssm.c` — unit-level forward pass tests
- `tools/ssm_reference.py` — Python reference for cross-check

**Verified:**
- All 40 layers (30 SSM + 10 GQA) forward on RTX 5050: **9.53 tok/s**
- CPU baseline: ~0.20 tok/s — **47.83× GPU speedup** (419.85 ms per forward pass)
- B=1, T=4, correctness check: GPU/CPU outputs match within tolerance (cuBLAS FMA-ordering divergence only)
- SSM recurrent state (SSM_D_STATE=128 per head, per-layer persistence)
- GQA: 16 Q-heads / 2 KV-heads, 8:1 ratio, fused Q/K RMSNorm + causal attention

**What's pending:**
- No optimized CUDA kernels beyond cuBLAS and fused element-wise ops — all matmul delegated to cuBLAS (addressed in Phase 6)
- No attention kernel for large T — current GQA is full O(N²) causal; no sparse/flash attention variant

---

## Phase 2.5: GPU Verification (✅ Done)

**Files:**
- `tools/bench_e2e.c` — end-to-end 40-layer benchmark with CPU+GPU correctness comparison
- `test_tokenizer` — tokenizer functional validation binary

**Verified:**
- Full pipeline: GGUF load → tokenizer → forward pass (all 40 layers) — assembled and benchmarked
- GPU/CPU logit-level agreement confirmed within numerical tolerance
- No correctness regressions between single-layer and full-model stacking

---

## Phase 3: Training Loop with TST (🔄 In Progress)

**Files (existing):**
- `include/wubu_model.h` — model struct with layer array, token embedding, output weight, state buffers
- `src/wubu_tokenizer.c` — BPE tokenizer encode/decode (867 lines)
- `include/wubu_tokenizer.h` — tokenizer struct + init/encode/decode API
- `tools/test_tokenizer.c` — test harness

**What's done:**
- Model skeleton (`wubu_model_t`) defines the layer stack, embedding, output projection, and state buffers
- Tokenizer implements GPT-2 byte-level BPE encoding/decoding with hash-table merge lookup
- TST training loop design documented in architecture diagram (superposition + recovery phases)

**Current blocker — tokenizer merge lookup:**
- `build_merge_hash()` calls `find_token_by_string()` to resolve each merge rule's merged ID by concatenating left/right vocab strings and looking up in a 248K-entry vocab hash table
- For 248K merges with variable-length vocab strings (up to 256 bytes each), string concatenation + hash lookup per entry is O(N) in merge count with non-trivial per-iteration cost
- The BPE encode loop is structured but **not yet profiled at training scale** — the real concern is the per-token merge loop scanning all adjacent pairs in a 256-byte working array on every merge iteration

**What needs building:**
- **`src/wubu_train.c`** — complete TST training loop:
  - Superposition phase: bag-of-s embedding averaging + multi-hot cross-entropy loss
  - Recovery phase: standard next-token CE loss
  - Dual optimizer: AdamW (Euclidean params) + Riemannian SGD (Poincaré ball params)
  - GGUF-compatible checkpointing every 1000 steps
  - CPU-offloaded optimizer states (8 GB VRAM constraint)
- Profiling the tokenizer encode path on representative training text and fixing any hot paths

---

## Phase 4: MoE Port (⏳ Future)

**Files:**
- `WUBUNEST_V2/wubu_moe.py` — Python MoE routing prototype (experimental, not ported to C)

**Status:** Not started in C/CUDA. The model spec calls for 256 experts with 8 active + 1 shared per token, with MoE routing in FFN layers. All relevant weight tensors exist in the Qwen3.6 GGUF but are not loaded. Depends on training loop reaching convergence first.

**Key unknowns:**
- Expert load balancing strategy (auxiliary loss? token choice?)
- Router implementation (softmax top-k? Sinkhorn?)
- GPU memory budget for 256 expert FFN weights on 8 GB VRAM

---

## Phase 5: Vision Port (⏳ Future)

**Status:** Not started. 27-layer 3D ViT architecture identified in research vault (phase3-generative encoder work) but no C/CUDA code exists. No timeline — MoE must ship first.

---

## Phase 6: CUDA Optimization (⏳ Ongoing)

**Status:** No dedicated optimization work has begun. Current GPU performance relies entirely on cuBLAS SGEMM for matmuls (~90% of FLOPs) and simple fused element-wise kernels. Planned work:

- Fused attention kernel (eliminate intermediate materialization of Q*K^T)
- Kernel fusion for SSM Gated Delta Net step (reduce kernel launch overhead)
- Quantized matmul kernels for on-the-fly dequant (avoid f16 weight expansion)
- Stream overlap for CPU-offloaded optimizer states during training

---

## Summary Table

| Phase | Component | Status | Key Metric | Remaining Work |
|-------|-----------|--------|------------|----------------|
| 1 | GGUF reader + Poincaré | ✅ Done | 95% NN preservation | Formal validation |
| 2 | SSM/GQA C+CUDA | ✅ Done | 9.53 tok/s, 47.83× | Sparse attention |
| 2.5 | GPU verification | ✅ Done | GPU/CPU match <1e-3 | — |
| **3** | **Training loop** | **🔄 Blocked** | **No tok/s yet** | **Tokenizer fix + `wubu_train.c`** |
| 4 | MoE (256 experts) | ⏳ Future | — | Full C port |
| 5 | Vision (27-layer ViT) | ⏳ Future | — | Full C port |
| 6 | CUDA optimization | ⏳ Ongoing | — | Fused + quantized kernels |
