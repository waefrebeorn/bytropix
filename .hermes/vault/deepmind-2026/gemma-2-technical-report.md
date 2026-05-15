# Gemma 2 Technical Report

**Title**: Gemma 2: Improving Open Language Models at a Practical Size
**Authors**: Gemma Team (Google DeepMind)
**Date**: 2024-07-31
**ArXiv**: https://arxiv.org/abs/2408.00118
**PDF**: https://arxiv.org/pdf/2408.00118

## Abstract

We introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging from 2B to 27B parameters. We apply several modifications to the Transformer architecture: interleaving local-global attentions and group-query attention (GQA). We train the 2B and 9B models with knowledge distillation instead of next token prediction. The resulting models deliver the best performance for their size.

## Key Architectural Findings

### 1. Interleaved Local-Global Attention
- Alternates between local (sliding window) and global (full) attention layers
- Local layers: efficient, limited receptive field
- Global layers: capture long-range dependencies
- **Relevance to WuBuText**: This is the direct predecessor to Gemma 3's approach and conceptually maps to our SSM (local) + GQA (global) design.

### 2. Grouped-Query Attention (GQA) — DIRECTLY USED
- Gemma 2 explicitly adopts GQA for efficient inference
- Multiple query heads share the same key/value heads
- Reduces KV cache size without proportional quality loss
- **Relevance to WuBuText**: Our 10 GQA layers follow the exact same GQA pattern. The implementation details from Gemma 2 guide our `attention.c` code.

### 3. Knowledge Distillation
- 2B and 9B models trained via distillation, not next-token prediction
- Teacher model's full output distribution used as training target
- Significantly improves small model performance
- **Relevance**: Potential training methodology for WuBuText if we distill from a larger model.

### 4. Model Sizes
- Gemma 2: 2B, 9B, 27B parameters
- All are dense (non-MoE) models

## Relevance to WuBuText AI

| Aspect | Gemma 2 | WuBuText |
|--------|---------|----------|
| GQA | Explicitly adopted | Our 10 GQA layers use same concept |
| Local/Global | Interleaved attention layers | SSM layers = local, GQA layers = global |
| KV Cache Benefit | Reduced via GQA | Further reduced via SSM in our design |

## References
- Gemma 2: https://arxiv.org/abs/2408.00118
- GQA paper: https://arxiv.org/abs/2305.13245
