# Fast Attention Research — Efficiency Techniques

Source papers: `.hermes/research/papers/research_papers/`

## Papers

| Paper | ID | Key Idea | For WuBu |
|-------|----|----------|----------|
| **NSA (Native Sparse Attention)** | 2503.10488 | DeepSeek's sparse attention: coarse + fine-grained token selection | Sparse attention for 1M+ ctx |
| **FlashAttention** | 2205.14135 | IO-aware exact attention, tiling | CUDA kernel pattern |
| **FlashAttention-2** | 2307.08691 | Better work partitioning | Production attention kernel |
| **Mamba2** | 2311.10763 | Selective SSM, hardware-efficient | Structured state space alternative |
| **MISA** | 2505.04888 | Mixture of Sparse Attention | Adaptive sparse patterns |
| **Gated Sparse Attention** | 2503.09542 | Learnable sparsity gates | Gating mechanism for sparse routing |
| **Delta Attention** | 2502.14864 | Delta rule for attention | Related to Gated DeltaNet |
| **Hyena Hierarchy** | 2312.12654 | Long convolutions > attention | O(n) sequence modeling |
| **C-Mamba** | 2504.05591 | Mamba in pure C | Implementation patterns in C |
| **StreamIndex** | 2505.04968 | Streaming context with index | Long context management |
| **MergeAttention** | 2503.08680 | Merge token clusters | Token clustering for wubu routing |
| **TokenFormer** | 2412.17950 | Token-token attention as RNN | Alternative attention formulation |

## What This Means for WuBu

1. **Gated DeltaNet (Qwen3.5+)** IS a form of linear attention related to Delta Attention.
   We can implement it directly — no need for full FlashAttention if 75% of layers are linear.

2. **NSA** is DeepSeek's approach to sparse attention for 1M+ context. Use this for the 25% full attention layers when scaling beyond 262K.

3. **Mamba2** is an alternative to attention for the linear part — we could replace Gated DeltaNet with Mamba2 if hyperbolic gyration doesn't work. But Gated DeltaNet's recurrence is simpler for hyperbolic adaptation.

4. **FlashAttention** is only needed for the 25% GQA layers. Those can use standard softmax attention.

5. **StreamIndex** technique for managing KV cache at 1M+ context — relevant when we scale.
