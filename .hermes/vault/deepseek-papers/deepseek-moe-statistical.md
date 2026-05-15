# On DeepSeekMoE: Statistical Benefits (2025)

**Title**: On DeepSeekMoE: Statistical Benefits of Shared Experts and Normalized Sigmoid Gating
**Authors**: Multiple (published May 2025)
**Date**: 2025-05-16
**ArXiv**: https://arxiv.org/abs/2505.10860
**PDF**: https://arxiv.org/pdf/2505.10860

## Abstract

Mixture of experts (MoE) methods are a key component in most large language model architectures, including the recent series of DeepSeek models. Compared to other MoE implementations, DeepSeekMoE stands out because of two unique features: the deployment of a shared expert strategy and of the normalized sigmoid gating. This paper provides a theoretical analysis of these features, demonstrating their statistical benefits.

## Key Findings

### 1. Theoretical Justification for Shared Experts
- Shared experts reduce redundancy in routed experts
- Statistical analysis shows shared experts capture dataset-level common patterns
- Routed experts can focus entirely on specialized knowledge without wasting capacity on common features

### 2. Normalized Sigmoid vs Softmax Gating
- Softmax gating creates competitive dynamics: experts compete for tokens
- Sigmoid gating: each expert's activation is independent — no competition
- Normalization after selection ensures proper weighting
- The independence of sigmoid allows experts to specialize more cleanly

## Relevance to WuBuText
Validates the architectural choices we've made for our MoE implementation. Papers like this provide theoretical grounding for why our approach works.

## References
- https://arxiv.org/abs/2505.10860
