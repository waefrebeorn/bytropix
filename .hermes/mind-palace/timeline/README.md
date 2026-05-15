# Timeline — WuBuText AI Actual Build History (May 2026)

All phases complete. Actual timeline vs original plan below.

## Actual Timeline

| Phase | Component | Target Week | Actual | Key Milestone |
|-------|-----------|-------------|--------|---------------|
| 0 | GGUF Tensor Layout | — | May 12 | 733 tensors, 13 GGML types parsed |
| 1 | Embedding Graft | Week 1 | May 12 | 95% NN preservation, R=0.956 |
| 2 | Attention Port | Week 2 | May 12-13 | 30 SSM + 10 GQA layers, CPU/GPU |
| 3 | Training Loop | Week 3 | May 13 | 177s/step baseline |
| 4 | MoE Port | Week 4 | May 13-14 | 256 expert, lazy dequant 9× |
| 5 | Vision Port | Week 6 | May 14 | 27-layer 3D ViT, 99ms GPU |
| 6 | CUDA Optimization | — | May 14-15 | 177s→11s/step (16×) |

## Key Dates
- **May 12:** Embeddings grafted, embedding→Poincaré verified
- **May 13:** Full 40-layer model forward working, backward passes verified
- **May 14:** All 7 cold gaps closed. NaN root cause found: gguf_raw_size(IQ2_XXS) 72→66. Per-expert dequant implemented. 177s→13s/step.
- **May 15:** v6 milestone. 11s/step, 0 NaN all flags. Vault audit done (12 vaults + tailslayer). Paper audit (32 Qwen files, 14 discrepancies catalogued). All SVGs updated.

## Remaining Priorities
- P0: GPU MoE forward (eliminate 40 syncs, target <3s)
- P1: PGA LR tuning, multi-step convergence
- P2: MRoPE, sparse attention port, tailslayer spec-decode kernel, Q-Controller, Hamilton encoder
- P3: MTP head, vision encoder verification
