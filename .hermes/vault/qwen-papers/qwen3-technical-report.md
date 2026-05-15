# Qwen3 Technical Report

**Title**: Qwen3 Technical Report
**Authors**: Qwen Team (An Yang, Anfeng Li, Baosong Yang, et al.)
**Date**: 2025-05-14
**ArXiv**: https://arxiv.org/abs/2505.09388
**PDF**: https://arxiv.org/pdf/2505.09388

## Abstract

We present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of LLMs designed to advance performance, efficiency, and multilingual capabilities. The series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models and enables dynamic mode switching based on user queries or chat templates. Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference. Empirical evaluations demonstrate state-of-the-art results across diverse benchmarks. Qwen3 expands multilingual support from 29 to 119 languages. All Qwen3 models are publicly accessible under Apache 2.0.

## Key Architectural Findings

### 1. MoE Architecture
- Qwen3 uses both dense and MoE variants, with the largest MoE model at 235B total parameters
- Architecture follows DeepSeekMoE style with fine-grained expert segmentation and shared experts
- Top-K routing with normalized sigmoid gating (per DeepSeekMoE findings)
- **Relevance to WuBuText**: Our 256 experts with 8 active per token follows the same fine-grained MoE philosophy. The normalized sigmoid gating function reduces routing collapse.

### 2. Thinking/Non-Thinking Mode
- Unified model architecture supporting both modes
- Mode controlled via chat template tokens (no separate model weights needed)
- Thinking budget mechanism: `max_thinking_tokens` parameter controls reasoning depth
- **Relevance to WuBuText**: Could be implemented as a runtime flag in C inference engine, controlling whether SSM/GQA layers do extra thinking passes or direct generation.

### 3. Architecture Configuration
- Dense models: 0.6B, 1.7B, 4B, 8B, 14B, 32B parameters
- MoE models: 30B-A3B (30B total, 3B active), 60B-A3B, 128B-A8B, 235B-A21B
- MoE models use fine-grained experts with shared expert isolation
- **Relevance to WuBuText**: Our 30 SSM + 10 GQA layering with 256 experts is a novel hybrid not seen in Qwen3, but the expert count and active ratio patterns are informative.

### 4. Long Context
- Support for up to 131,072 tokens (128K) context
- Uses sliding window attention for efficiency in long sequences
- **Relevance to WuBuText**: WuBuText targets 256K context, double Qwen3's base. The sparse/chunked prefill methods from Qwen2.5-1M become critical.

### 5. Training Details
- Pre-training: 18.5T tokens for small models, up to 36.6T for large models
- Uses knowledge distillation from larger models to smaller ones
- SFT + RL post-training pipeline

## Relevance to WuBuText AI

| Aspect | Qwen3 Finding | WuBuText Application |
|--------|---------------|---------------------|
| MoE Scale | 235B total, 21B active | Our 256 experts, 8 active is different granularity |
| Hybrid Architecture | Dense vs MoE models separate | Our SSM+GQA hybrid within single model is novel |
| Thinking Mode | Token-controlled mode switch | Add as C runtime flag for inference path selection |
| Expert Gating | Normalized sigmoid from DeepSeekMoE | Implement in `moe.c` gating function |
| Long Context | 128K via sliding window | Need own 256K approach (sparse attention) |

## Key Equations / Mechanisms

**Expert Gating** (from DeepSeekMoE, used in Qwen3):
```
g_i = sigmoid(s_i) / sum(sigmoid(s_j) for j in top-K)
```
where s_i is the router logit for expert i.

**Thinking Budget**:
```
max_thinking_tokens = min(config.max_thinking_tokens, query_complexity * scale_factor)
```

## References
- Qwen3 Paper: https://arxiv.org/abs/2505.09388
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
