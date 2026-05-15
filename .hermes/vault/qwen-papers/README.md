# Qwen Papers Vault — WuBuText AI Architectural Insights

This vault contains extracted architectural insights from Qwen research papers (2024-2026) relevant to WuBuText AI's C+CUDA implementation of Qwen3.6 architecture.

## Architecture Overview for WuBuText

WuBuText AI implements: **30 SSM layers + 10 GQA layers, 256 experts MoE with 8 active, 256K context**

Key architectural themes traced across Qwen papers:
- Hybrid attention mechanisms (dense + MoE variants)
- Thinking vs non-thinking mode switching
- Long-context scaling (128K → 1M tokens)
- MoE routing and expert specialization
- KV-cache optimization for long context

---

## Paper Index

### 1. Qwen3 Technical Report (2505.09388)
- **Date**: 2025-05-14
- **Link**: https://arxiv.org/abs/2505.09388
- **File**: [qwen3-technical-report.md](qwen3-technical-report.md)

### 2. Qwen2.5 Technical Report (2412.15115)
- **Date**: 2024-12-19
- **Link**: https://arxiv.org/abs/2412.15115
- **File**: [qwen2.5-technical-report.md](qwen2.5-technical-report.md)

### 3. Qwen2.5-1M Technical Report — Long Context (2501.15383)
- **Date**: 2025-01-26
- **Link**: https://arxiv.org/abs/2501.15383
- **File**: [qwen2.5-1m-technical-report.md](qwen2.5-1m-technical-report.md)

### 4. Qwen3.5-Omni Technical Report (2604.15804)
- **Date**: 2026-04-17
- **Link**: https://arxiv.org/abs/2604.15804
- **File**: [qwen3.5-omni-technical-report.md](qwen3.5-omni-technical-report.md)

### 5. Qwen-Image-2.0 Technical Report (2605.10730)
- **Date**: 2026-05-11
- **Link**: https://arxiv.org/abs/2605.10730
- **File**: [qwen-image-2.0-technical-report.md](qwen-image-2.0-technical-report.md)

### 6. Qwen3-VL Technical Report (2511.21631)
- **Date**: 2025-11-26
- **Link**: https://arxiv.org/abs/2511.21631
- **File**: [qwen3-vl-technical-report.md](qwen3-vl-technical-report.md)

### 7. Qwen3-Omni Technical Report (2509.17765)
- **Date**: 2025-09-22
- **Link**: https://arxiv.org/abs/2509.17765
- **File**: [qwen3-omni-technical-report.md](qwen3-omni-technical-report.md)

---

## Key Architectural Findings Summary

| Topic | Finding | Relevance to WuBuText |
|-------|---------|----------------------|
| **MoE Routing** | Qwen3 uses fine-grained MoE with shared experts per DeepSeekMoE-style design | Directly informs our 256-expert, 8-active routing |
| **Think/Non-Think** | Unified model with dynamic mode switching via chat template | Could inspire inference-mode selection in C impl |
| **Long Context** | Qwen2.5-1M achieves 1M context via progressive pretraining + sparse attention + chunked prefill | 4x extrapolation method directly applicable |
| **Hybrid Attention** | Qwen3.5-Omni uses Hybrid Attention MoE for both Thinker and Talker | SSM + GQA hybrid already in WuBuText |
| **KV Cache** | Gemma 3 insight: increase local:global attention ratio to reduce KV cache | Relevant for our 256K context GQA layers |
| **Speculative Decoding** | Not directly from Qwen papers; see DeepSeek vault | Cross-reference needed |
