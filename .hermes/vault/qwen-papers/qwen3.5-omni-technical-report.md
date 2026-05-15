# Qwen3.5-Omni Technical Report

**Title**: Qwen3.5-Omni Technical Report
**Authors**: Qwen Team
**Date**: 2026-04-17
**ArXiv**: https://arxiv.org/abs/2604.15804
**PDF**: https://arxiv.org/pdf/2604.15804

## Abstract

We present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. It scales to hundreds of billions of parameters and supports a 256K context length. By leveraging a massive dataset of heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model supports over 10 hours of audio understanding and 400 seconds of 720P video. It introduces ARIA for dynamic alignment of text and speech units, enhancing stability and prosody.

## Key Architectural Findings

### 1. Hybrid Attention MoE Framework — CRITICAL FOR WUBUTEXT
- **Both Thinker and Talker components use Hybrid Attention MoE**
- This is the closest described architecture to WuBuText's SSM + GQA hybrid found in published literature
- Enables efficient long-sequence inference (256K context)
- **Relevance to WuBuText**: Validates our architectural choice of hybrid attention (30 SSM layers + 10 GQA layers) as a viable path for efficient long-context modeling.

### 2. 256K Context Length
- Directly matches WuBuText's target context length of 256K
- Achieved through the Hybrid Attention MoE design
- **Relevance**: Confirms 256K is achievable with this architectural family.

### 3. ARIA (Alignment of Representation for Interactive Audio)
- Dynamically aligns text and speech unit encoding
- Addresses instability in streaming speech synthesis
- While primarily for speech, the concept of dynamic modality alignment could inform cross-modal routing in our architecture.

### 4. Model Scale
- Hundreds of billions of parameters
- Massive training dataset (heterogeneous text-vision + 100M hours audio-visual)

## Relevance to WuBuText AI

This is the closest published architecture to what WuBuText is building. Key takeaways:

| Aspect | Qwen3.5-Omni | WuBuText |
|--------|-------------|----------|
| Architecture | Hybrid Attention MoE | 30 SSM + 10 GQA + 256 Expert MoE |
| Context | 256K | 256K (same target!) |
| MoE | Yes, in Thinker + Talker | Yes, 256 experts, 8 active |
| Attention | Hybrid (likely local + global) | GQA in 10 layers, SSM in 30 layers |
| Validation | Published SOTA model | Confirms our architectural direction |

## References
- Qwen3.5-Omni: https://arxiv.org/abs/2604.15804
