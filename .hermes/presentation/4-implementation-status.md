# 4. Implementation Status (May 15 PM v6)

**All phases complete. Training integrated at 11s/step (16×). Zero NaN.**

---

## Phase 0: GGUF Reader ✅
- Full GGUF format parsing (13 GGML types)
- 733 tensors from Qwen3.6
- GPU weight loading fixed (dequant bypass in unbuffered reads)
- **gguf_raw_size(IQ2_XXS) fixed: 72→66 bytes/block** — was wrong, caused NaN cascade

## Phase 1: Embedding Graft ✅
- Euclidean → Poincaré exponential mapping (R=0.956)
- ~95% nearest-neighbor preservation
- Embeddings: 2.03GB, 248K tokens

## Phase 2: SSM/GQA Forward Pass ✅
- All 40 layers (30 SSM + 10 GQA) CPU/GPU forward
- GPU weight loading match verified
- TGT NaN/Inf guards applied everywhere (GPU CUDA kernels + CPU)
- CUDA kernels: matmul, SiLU, sigmoid, softplus, RMSNorm, delta_net_step

## Phase 3: Training Loop ✅
**Integrated in train_integrated binary. 11s/step. 0 NaN.**

| Feature | Flag | Status |
|---------|------|--------|
| Per-expert IQ2_XXS dequant | — | **177s→11s/step (16×)** |
| GPU output projection | — | cublasSgemm replaces 2B CPU FMAs |
| Async D→H copies | — | PGA-only arrays skipped when !pga |
| Persistent MoE buffers | — | No per-step 3GB alloc/free |
| RSGD optimizer | RSGD=1 | ✅ Verified |
| Poincaré GQA | PGA=1 | ✅ Verified (LR issue noted) |
| Nested SSM K=4 | NESTED_SSM=1 | ✅ Verified |
| TST Training | TST=1 | ✅ Verified |
| Nested MoE | NESTED_MOE=1 | ✅ Verified |
| Hyperbolic SSM | POINCARE_R=x | ✅ Verified |
| All 6 flags combined | All | ✅ Verified, 0 NaN |

**All 7 cold gaps closed:**
- ✅ Poincaré GQA backward — dQ=1.95, dK=0.004, dV=0.70, dX=571
- ✅ Nested SSM backward — K=1,2,3 all pass, 0 NaN
- ✅ Möbius linear layer (M⊗) — output in ball, fwd+bwd verified
- ✅ Gyration closed-form — exact match, ~3× faster
- ✅ Hyperbolic output projection — 5/5 pass
- ✅ Nested MoE 2-level backward — 15/15 tests pass
- ✅ Hyperbolic KV cache — prefill + incremental verified

## Phase 4: MoE Port ✅
- Per-expert IQ2_XXS dequant: 3.9ms/expert, top-8 only
- Transpose: raw[ff][model] → [model][ff] for moe_expert_forward
- Persistent buffers in lmoe_t (no per-step 3GB alloc/free)
- Hidden max=13 (was 5e9 from buggy strided extraction)

## Phase 5: Vision Port ✅
- 27-layer 3D ViT GPU: 99ms (128×128), 0 NaN
- Vision→text pipeline: real screenshot, 0 NaN

## Phase 6: CUDA Kernels ✅
- cuBLAS matmul for all projections
- SSM scan kernel (parallel associative prefix)
- MoE dispatch kernel (grouped per expert)
- GQA attention kernel
- All pass max_diff < 6e-8
- `tgt_safe_expf` in 4 GPU kernel sites

## Remaining

| Issue | Severity | Detail |
|-------|----------|--------|
| ~11s/step GPU compute | Performance | 40 layers × ~275ms on RTX 5050 |
| PGA loss jumps 21.6→69 | Numeric | LR too high for PGA backward |
| CONV_DIM 8192 vs config 1536 | Correctness | Possible SSM layernorm/conv discrepancy |
| MRoPE 3D missing | Correctness | Position encoding degrades at >32K |
| MTP head missing | Feature | Multi-token prediction |
| 13 vaults + tailslayer unported | Scope | Python/JAX/C++ prototypes waiting for C |
| 3 new SVGs | Diagrams | training-pipeline, tailslayer-pattern, paper-audit |
