# Qwen2.5 Technical Report

**Title**: Qwen2.5 Technical Report
**Authors**: Qwen Team (An Yang, Baosong Yang, et al.)
**Date**: 2024-12-19
**ArXiv**: https://arxiv.org/abs/2412.15115
**PDF**: https://arxiv.org/pdf/2412.15115

## Abstract

We present Qwen2.5, a comprehensive upgrade of the Qwen model family. Qwen2.5 includes dense language models ranging from 0.5B to 72B parameters, and Mixture-of-Experts (MoE) models up to 236B parameters. The flagship MoE model, Qwen2.5-MoE, adopts an advanced architecture with fine-grained experts, shared experts, and routed experts. We pre-trained the models on up to 18 trillion tokens with improved data quality. The models demonstrate strong performance on benchmarks covering reasoning, coding, mathematics, multilingual tasks, and agentic capabilities.

## Key Architectural Findings

### 1. Baseline Architecture for Qwen3
- Qwen2.5 is the direct predecessor to Qwen3
- Dense models use standard Transformer decoder architecture
- MoE models use fine-grained experts + shared experts
- Supports 128K context length at base

### 2. Qwen2.5-MoE Architecture
- Uses similar design to DeepSeekMoE: fine-grained expert segmentation
- Shared experts capture common knowledge across all tokens
- Routed experts specialize in different domains
- Normalized sigmoid gating for expert selection
- **Relevance to WuBuText**: Establishes the architectural lineage that leads to our target Qwen3.6. The shared + routed expert pattern is implemented in our MoE routing code.

### 3. Model Sizes
- Dense: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- MoE: 14B-A2.7B, 47B-A4B, 140B-A14B, 236B-A21B
- **Relevance to WuBuText**: Our architecture (30 SSM + 10 GQA, 256 experts) doesn't directly match any Qwen2.5 config. We are building a novel hybrid.

### 4. Training
- 18T tokens pre-training
- Improved data quality and mixture
- SFT + RL (GRPO) for alignment
- **Relevance**: GRPO (Group Relative Policy Optimization) is relevant for any RL-based fine-tuning we might do later.

## Relevance to WuBuText AI

This paper establishes the foundational architecture that Qwen3 builds upon. Key takeaways for WuBuText:
- The MoE design (shared + routed experts with sigmoid gating) is confirmed as effective
- Context window of 128K is baseline; WuBuText targets 256K (doubling)
- The progression from Qwen2.5 → Qwen3 gives architectural context for our Qwen3.6 target

## References
- Qwen2.5: https://arxiv.org/abs/2412.15115
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
