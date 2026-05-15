# DeepSeek-V3.2 Technical Report

**Title**: DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
**Authors**: DeepSeek-AI
**Date**: 2025-12-02
**ArXiv**: https://arxiv.org/abs/2512.02556
**PDF**: https://arxiv.org/pdf/2512.02556

## Abstract

We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. Key breakthroughs: (1) DeepSeek Sparse Attention (DSA) — an efficient attention mechanism that reduces computational complexity while preserving performance in long-context scenarios; (2) Scalable Reinforcement Learning Framework — DeepSeek-V3.2 performs comparably to GPT-5, with the high-compute variant DeepSeek-V3.2-Speciale surpassing GPT-5 and matching Gemini-3.0-Pro, achieving gold-medal performance in IMO 2025 and IOI 2025; (3) Large-Scale Agentic Task Synthesis Pipeline for integrating reasoning into tool-use scenarios.

## Key Architectural Findings

### 1. DeepSeek Sparse Attention (DSA) — CRITICAL for Long Context
- **What**: A sparse attention pattern that reduces computational complexity from O(L²) to O(L log L) or O(L)
- Designed specifically for long-context scenarios
- Preserves model quality despite sparsity
- Can be combined with MLA for maximum efficiency
- **Relevance to WuBuText**: DSA provides another attention optimization for our GQA layers. Combined with SSM's linear complexity, DSA in GQA layers would give us full linear-time inference at 256K context.

### 2. Reinforcement Learning at Scale
- Robust RL protocol that scales with compute
- Achieves GPT-5-comparable performance
- IMO and IOI gold medal performance
- Shows that RL post-training is critical for reasoning

### 3. Agentic Task Synthesis
- Novel pipeline for generating agentic training data at scale
- Improves instruction-following in complex interactive environments

## Relevance to WuBuText AI

| Aspect | DSA in V3.2 | WuBuText Application |
|--------|-------------|---------------------|
| Sparse Attention | O(L log L) complexity for long context | Apply to GQA layers for 256K efficiency |
| RL Training | Scalable RL protocol | Reference for post-training phase |
| Agent Tasks | Synthesis pipeline for tool-use | Future consideration for agent capabilities |

## Key Equations

**DSA Pattern** (conceptual):
For each query position i, attend to a sparse set of positions:
```
S(i) = {i} ∪ local_window(i, w) ∪ global_positions(i, g)
```
where w is local window size and g is number of global positions.

## References
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
- DeepSeek Sparse Attention: detailed in V3.2 paper
