# Qwen2.5-1M Technical Report — Long Context Breakthrough

**Title**: Qwen2.5-1M Technical Report
**Authors**: Qwen Team (An Yang, Bowen Yu, et al.)
**Date**: 2025-01-26
**ArXiv**: https://arxiv.org/abs/2501.15383
**PDF**: https://arxiv.org/pdf/2501.15383

## Abstract

We introduce Qwen2.5-1M, a series of models that extend the context length to 1 million tokens. Compared to the previous 128K version, the Qwen2.5-1M series have significantly enhanced long-context capabilities through long-context pre-training and post-training. Key techniques such as long data synthesis, progressive pre-training, and multi-stage supervised fine-tuning are employed. We present an inference framework including a length extrapolation method that can expand model context lengths by at least 4x without additional training. To reduce inference costs, we implement sparse attention with chunked prefill optimization and sparsity refinement to improve precision. The Qwen2.5-1M models achieve a remarkable 3x to 7x prefill speedup in scenarios with 1 million tokens of context.

## Key Architectural Findings

### 1. Length Extrapolation Method (Critical for WuBuText 256K)
- Can extend context by at least 4x without additional training
- Uses position interpolation techniques to extend RoPE beyond trained length
- **Implementation**: Modifies the frequency scaling of RoPE embeddings
- **Relevance to WuBuText**: If our base model trains at 64K, this method can extrapolate to 256K without retraining. Directly applicable to our `attention.c` RoPE implementation.

### 2. Sparse Attention with Chunked Prefill
- Sparse attention reduces the quadratic complexity of full attention
- Chunked prefill processes long prompts in batches for better memory efficiency
- **Key equation**: For sequence length L, sparse attention reduces from O(L²) to O(L × window) where window << L
- **Relevance to WuBuText**: This is exactly what our GQA layers need for 256K context. Combine with SSM's linear complexity for the full benefit.

### 3. Progressive Pre-training
- Start with 4K context, progressively extend to 32K, 128K, 1M
- Each stage adds more long-range training data
- **Relevance to WuBuText**: Training strategy for our model: start with shorter contexts and gradually extend.

### 4. Inference Framework Optimizations
- Kernel optimization for sparse attention
- Pipeline parallelism for long sequences
- Scheduling optimization to overlap compute and I/O
- **Relevance**: All applicable to our C inference engine.

### 5. Model Sizes
- Qwen2.5-7B-Instruct-1M (7B params)
- Qwen2.5-14B-Instruct-1M (14B params)
- Qwen2.5-Turbo (API-accessed)
- 14B model significantly outperforms GPT-4o-mini in long-context tasks

## Relevance to WuBuText AI — CRITICAL

This is one of the most important papers for WuBuText's long-context goal (256K):

| Technique | Qwen2.5-1M Approach | WuBuText Application |
|-----------|---------------------|---------------------|
| Position Extrapolation | 4x RoPE scaling without retraining | Implement in `attention.c` rope function |
| Sparse Attention | Chunked prefill + sparse pattern | Combine with SSM for GQA layers |
| Progressive Training | 4K → 32K → 128K → 1M | Adopt for our training pipeline |
| KV Cache Management | Chunked prefill reduces peak memory | Critical for 256K context GQA layers |
| Prefill Speedup | 3x-7x improvement | Targets for our CUDA kernel optimizations |

## Key Equations

**RoPE Frequency Scaling for Extrapolation**:
```
θ_i = base^(-2i/d) × scale_factor
```
where scale_factor < 1 extends effective context length.

**Chunked Prefill**:
For prompt of length L, split into chunks of size C:
```
Prefill(1..L) = concat(Prefill(1..C), Prefill(C+1..2C), ..., Prefill(L-C+1..L))
```

## References
- Qwen2.5-1M: https://arxiv.org/abs/2501.15383
- RoPE: https://arxiv.org/abs/2104.09864
- YaRN (length extrapolation): https://arxiv.org/abs/2309.00071
