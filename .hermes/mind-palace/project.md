# WuBuText AI — Project Overview (May 15 PM v6)

## Mission
Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry.
**All phases complete.** Training at 11s/step (16× improvement). Zero NaN all configs.

## All Phases Complete ✅

| Phase | Component | Status | Key Metric |
|-------|-----------|--------|------------|
| 0 | GGUF Tensor Layout | ✅ | 733 tensors, 13 types |
| 1 | Embedding Graft | ✅ | 95% NN preservation, R=0.956 |
| 2 | Attention Port (SSM+GQA) | ✅ | 30 SSM + 10 GQA layers, CPU/GPU |
| 3 | Training Loop | ✅ | 11s/step, CE 21.6→18.4, 0 NaN |
| 4 | MoE Port | ✅ | 256 expert, lazy dequant 9×, persistent buffers |
| 5 | Vision Port | ✅ | 27-layer 3D ViT, 99ms GPU, 0 NaN |
| 6 | CUDA Optimization | ✅ | SSM scan + MoE dispatch, cublas proj |

## Key Achievements

- **gguf_raw_size(IQ2_XXS) fix**: 72→66 bytes/block — eliminated NaN cascade
- **Per-expert IQ2_XXS dequant**: 3.9ms/expert vs 3GB full dequant — **177s→11s/step (16×)**
- **GPU output projection**: cublasSgemm replaces 2B CPU FMAs (V=248320, D=2048)
- **7 cold gaps all closed**: Every backward pass verified (Poincaré GQA, Nested SSM K=1/2/3, M⊗, gyration, hyperbolic output proj, MoE 2-level, hyperbolic KV cache)
- **6 env flags all verified**: TST/RSGD/PGA/NSSM/NMOE/POINCARE_R — individually + combined, 0 NaN
- **GPU vision**: Full 27-layer 3D ViT at 99ms/128×128, text-pipeline integrated

## Remaining

| Issue | Severity |
|-------|----------|
| ~11s/step GPU compute bound (40 layers SSM/GQA on RTX 5050) | Performance |
| PGA loss jump (21.6→69) | Numeric — LR too high for PGA backward |
| CONV_DIM=8192 vs config 1536 | Possible SSM layernorm/conv discrepancy |
| MRoPE 3D not implemented | Position encoding degrades at >32K |
| MTP prediction head missing | 1-layer future token prediction |
| 12 vaults with unported theory | Python/JAX prototypes waiting for C port |
| **Tailslayer spec-decode kernel (new May 15)** | Hedged-read CUDA: N drafts, first-valid-wins |
| Sliding window pair sampling for draft-target alignment | P2 |
