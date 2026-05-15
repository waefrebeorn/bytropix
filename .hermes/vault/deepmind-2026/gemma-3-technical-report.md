# Gemma 3 Technical Report

**Title**: Gemma 3 Technical Report
**Authors**: Gemma Team (Google DeepMind)
**Date**: 2025-03-25
**ArXiv**: https://arxiv.org/abs/2503.19786
**PDF**: https://arxiv.org/pdf/2503.19786

## Abstract

We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging from 1 to 27 billion parameters. This version introduces vision understanding abilities, wider coverage of languages and longer context — at least 128K tokens. We change the architecture to reduce KV-cache memory by increasing the ratio of local to global attention layers, and keeping the span on local attention short. The models are trained with distillation and achieve superior performance to Gemma 2. Gemma3-27B-IT is comparable to Gemini-1.5-Pro across benchmarks. We release all models to the community.

## Key Architectural Findings

### 1. KV-Cache Reduction via Local/Global Attention Ratio — CRITICAL
- **Problem**: KV cache explodes with long context (128K+ tokens)
- **Solution**: Increase ratio of local attention layers to global attention layers
- **Implementation**: Most layers use sliding window (local) attention; few layers use global attention
- **Effect**: Local attention has O(L × w) KV cache vs global attention's O(L²)
- **Relevance to WuBuText**: This design principle directly applies to our hybrid architecture. Our 30 SSM layers are effectively "local" (linear complexity), while 10 GQA layers handle global interactions. Gemma 3 confirms this ratio approach works.

### 2. Long Context (128K tokens)
- Gemma 3 achieves at least 128K context
- Uses the local/global attention split to manage memory
- **Relevance**: Our 256K target is 2x Gemma 3, achievable with our SSM + GQA design.

### 3. Model Sizes
- Gemma 3: 1B, 4B, 12B, 27B parameters
- All multimodal (text + vision)

### 4. Knowledge Distillation Training
- Training with distillation from larger models
- Gemma3-4B-IT competitive with Gemma2-27B-IT
- Shows power of distillation for small model performance

## Relevance to WuBuText AI

| Aspect | Gemma 3 | WuBuText |
|--------|---------|----------|
| KV Cache Strategy | Local = short span, Global = full span | SSM = linear state, GQA = attention |
| Local/Global Ratio | More local layers than global | 30 SSM : 10 GQA = 3:1 ratio |
| Context Length | 128K tokens | 256K tokens (2x) |
| Architectural Insight | Local attention for efficiency | SSM replaces local attention entirely |

## Key Insight for WuBuText

Gemma 3's approach of "more local attention layers with short span + fewer global attention layers" is conceptually the same strategy as our "30 SSM layers + 10 GQA layers". The SSM serves the role of the local attention (efficient, bounded state) while GQA serves as global attention. **This validates our architectural choice from a major lab.**

## References
- Gemma 3: https://arxiv.org/abs/2503.19786
