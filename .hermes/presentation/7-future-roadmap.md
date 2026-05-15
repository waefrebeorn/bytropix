# 7. Future Roadmap (May 15 PM)

## Immediate: Integration Sprint
Wire all 7 standalone math modules into training pipeline:

| Module | Integration Target | Priority |
|--------|-------------------|----------|
| TST (bag+MCE) | train_gpu — replace CE with MCE | P0 |
| RSGD | train_gpu — optimizer for Poincaré params | P0 |
| Poincaré GQA | wubu_model_forward — replace GQA layers | P1 |
| Nested SSM K=4 | wubu_model_forward — replace SSM layers | P1 |
| Nested MoE | wubu_model_forward — replace router | P1 |
| Data pipeline | train_gpu — replace tokenizer input | P1 |
| CUDA kernels | train_gpu — replace CPU matmuls | P2 |

## Phase 7: Full Training
- Wire all 7 modules into single training binary
- Train on 1M+ token corpus
- Verify CE convergence < 5.0
- Add checkpointing (GGUF format save/load)
- Add gradient checkpointing for VRAM

## Phase 8: Validation
- Compare output vs Qwen3.6 reference forward pass
- Measure perplexity on WikiText-2
- Expert utilization entropy tracking
- Embedding norm stability monitoring

## Phase 9: Production
- Fix GPU vision pipeline timeout
- Fix 0.5% NaN bug
- Fix CPU RMSNorm OOB
- RSGD + AdamW dual optimizer
- Vision→text at scale
