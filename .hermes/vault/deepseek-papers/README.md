# DeepSeek Papers Vault — WuBuText AI Architectural Insights

This vault contains extracted architectural insights from DeepSeek research papers and blog posts (2024-2025) relevant to WuBuText AI's C+CUDA implementation.

## Architecture Overview for WuBuText

DeepSeek's contributions are foundational to WuBuText's architecture:
- **DeepSeekMoE**: Fine-grained experts + shared experts + normalized sigmoid gating → directly used in WuBuText's 256-expert MoE
- **Multi-head Latent Attention (MLA)**: KV-cache compression technique
- **DeepSeek Sparse Attention (DSA)**: Efficient attention for long context
- **Auxiliary-loss-free load balancing**: Training stability for MoE
- **Multi-token prediction (MTP)**: Training objective that improves speculative decoding

---

## Paper Index

### 1. DeepSeek-V3.2 Technical Report (2512.02556)
- **Date**: 2025-12-02
- **Link**: https://arxiv.org/abs/2512.02556
- **File**: [deepseek-v3.2-technical-report.md](deepseek-v3.2-technical-report.md)

### 2. DeepSeek-V3 Technical Report (2412.19437)
- **Date**: 2024-12-27
- **Link**: https://arxiv.org/abs/2412.19437
- **File**: [deepseek-v3-technical-report.md](deepseek-v3-technical-report.md)

### 3. DeepSeek-R1 Technical Report (2501.12948)
- **Date**: 2025-01-22
- **Link**: https://arxiv.org/abs/2501.12948
- **File**: [deepseek-r1-technical-report.md](deepseek-r1-technical-report.md)

### 4. DeepSeekMoE Architecture Paper (2401.06066)
- **Date**: 2024-01-11
- **Link**: https://arxiv.org/abs/2401.06066
- **File**: [deepseek-moe-architecture.md](deepseek-moe-architecture.md)

### 5. On DeepSeekMoE: Statistical Benefits (2505.10860)
- **Date**: 2025-05-16
- **Link**: https://arxiv.org/abs/2505.10860
- **File**: [deepseek-moe-statistical.md](deepseek-moe-statistical.md)

### 6. DeepSeek Blog Posts
- **Blog**: https://api-docs.deepseek.com/news/news
- **File**: [deepseek-blog-posts.md](deepseek-blog-posts.md)

---

## Key Architectural Findings Summary

| Topic | Finding | Relevance to WuBuText |
|-------|---------|----------------------|
| **MoE Design** | Fine-grained experts (mN) + shared experts (Ks) + sigmoid gating | Directly used: our 256 experts, 8 active |
| **Load Balancing** | Auxiliary-loss-free via dynamic bias adjustment | Implement in `moe.c` router |
| **MLA** | Low-rank KV projection reduces KV cache by ~75% | Alternative to our GQA approach |
| **DSA** | Sparse attention for long-context efficiency | Can supplement SSM in GQA layers |
| **MTP** | Multi-token prediction target (next D tokens) | Directly benefits speculative decoding |
| **R1 RL** | Pure RL without human reasoning traces | Training methodology for reasoning |
