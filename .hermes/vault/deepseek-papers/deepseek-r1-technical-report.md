# DeepSeek-R1: Reasoning via Reinforcement Learning

**Title**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Authors**: DeepSeek-AI (Daya Guo, Dejian Yang, et al.)
**Date**: 2025-01-22
**ArXiv**: https://arxiv.org/abs/2501.12948
**PDF**: https://arxiv.org/pdf/2501.12948

## Abstract

We show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. The trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields.

## Key Architectural Findings

### 1. Pure RL for Reasoning (No Human Reasoning Traces)
- LLMs can develop reasoning capabilities purely through RL, without SFT on human reasoning chains
- Emergent behaviors: self-reflection, verification, strategy adaptation
- **Relevance**: Training methodology for future reasoning enhancement of WuBuText

### 2. Distillation for Smaller Models
- Large model's reasoning patterns can be distilled into smaller models
- DeepSeek-R1-Distill variants (Qwen, Llama based) show strong reasoning
- **Relevance**: If we want a reasoning-enhanced WuBuText, distill from a larger reasoning model

### 3. Chain-of-Thought at Inference Time
- The RL-trained model naturally produces longer, more careful reasoning chains
- These chains show verification and backtracking behaviors

## Relevance to WuBuText AI

| Aspect | DeepSeek-R1 | WuBuText Application |
|--------|-------------|---------------------|
| RL Training | Pure RL without human trajectories | Reference for post-training pipeline |
| Reasoning Emergence | Self-reflection, verification | Could emerge in our model with RL training |
| Knowledge Distillation | From large reasoning model to small | Path to add reasoning to WuBuText |
| Inference Cost | Longer chains = more compute per token | Trade-off to manage in C impl |

## References
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
