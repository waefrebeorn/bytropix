# DeepMind 2025-2026 Papers Vault — WuBuText AI Architectural Insights

This vault contains extracted architectural insights from Google DeepMind research papers (2025-2026) relevant to WuBuText AI.

## Architecture Overview for WuBuText

DeepMind/Google contributes key architectural innovations in:
- **Gemma family**: Lightweight open models with local/global attention and GQA
- **KV-cache optimization**: Techniques for long-context efficiency
- **Knowledge distillation**: Training smaller models from larger ones
- **MoE and attention hybrids**: Emerging trends in efficient architectures

---

## Paper Index

### 1. Gemma 3 Technical Report (2503.19786)
- **Date**: 2025-03-25
- **Link**: https://arxiv.org/abs/2503.19786
- **File**: [gemma-3-technical-report.md](gemma-3-technical-report.md)

### 2. Gemma 2: Improving Open Language Models (2408.00118)
- **Date**: 2024-07-31 (updated 2025)
- **Link**: https://arxiv.org/abs/2408.00118
- **File**: [gemma-2-technical-report.md](gemma-2-technical-report.md)

### 3. Recurrent DeepMind Architectures (Griffin/Hawk)
- **Link**: https://arxiv.org/abs/2402.19427 (Griffin: Mixing Gated Linear Recurrences with Local Attention)
- **File**: [griffin-hawk-architecture.md](griffin-hawk-architecture.md)

---

## Key Architectural Findings Summary

| Topic | Finding | Relevance to WuBuText |
|-------|---------|----------------------|
| **Local/Global Attention** | Gemma 3 increases local:global ratio to reduce KV cache | Directly applicable to our GQA layer design |
| **GQA** | Gemma 2 uses Grouped-Query Attention | Our 10 GQA layers follow same pattern |
| **KV Cache Reduction** | Local attention = smaller KV cache | Critical for our 256K context target |
| **Knowledge Distillation** | Gemma 2: train 2B/9B with distillation not next-token | Training methodology reference |
| **Long Context** | Gemma 3: at least 128K tokens | Baseline for our 256K target |
| **SSM/Recurrence** | Griffin/Hawk: gated linear recurrences as attention alternative | SSM alternative — our 30 SSM layers use similar principle |

## Cross-References to Griffin/Hawk (2024)
While published in early 2024, Griffin/Hawk from DeepMind pioneered gated linear recurrence for language modeling — the direct predecessor to many SSM approaches. This is conceptually similar to our SSM layers.
