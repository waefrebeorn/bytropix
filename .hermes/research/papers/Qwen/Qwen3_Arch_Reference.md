# Qwen3 Architecture Reference

Extracted from "Qwen3 Technical Report" (arxiv:2505.09388)
Source: `Qwen3_Paper.pdf` / `Qwen3_Paper_Text.txt`

## Dense Models

All dense models use: **GQA** (Grouped Query Attention), **SwiGLU** activation, **RoPE** positional embeddings,
**RMSNorm** with pre-normalization. No QKV-bias (unlike Qwen2). **QK-Norm** for training stability.

| Model | Layers | Q Heads | KV Heads | Tie Embedding | Context |
|-------|--------|---------|----------|---------------|---------|
| Qwen3-0.6B | 28 | 16 | 8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16 | 8 | Yes | 32K |
| Qwen3-4B | 36 | 32 | 8 | Yes | 128K |
| Qwen3-8B | 36 | 32 | 8 | No | 128K |
| Qwen3-14B | 40 | 40 | 8 | No | 128K |
| Qwen3-32B | 64 | 64 | 8 | No | 128K |

## MoE Models

Same fundamental architecture as dense. Fine-grained expert segmentation.
**128 total experts, 8 activated per token.** No shared experts (unlike Qwen2.5-MoE).
Global-batch load balancing loss.

| Model | Layers | Q Heads | KV Heads | Total/Active Experts | Context |
|-------|--------|---------|----------|---------------------|---------|
| Qwen3-30B-A3B | 48 | 32 | 4 | 128/8 | 128K |
| Qwen3-235B-A22B | 94 | 64 | 4 | 128/8 | 128K |

## Tokenizer

- Byte-level BPE (BBPE)
- Vocabulary size: **151,669**
- Same tokenizer since Qwen1 (Bai et al., 2023)

## Key Implementation Notes for WuBu

1. **QK-Norm** applied to attention mechanism — applies LayerNorm/RMSNorm to Q and K before dot product
2. **GQA** with large Q:KV ratio (up to 8:1 for 32B model)
3. **Pre-normalization** — RMSNorm before attention and FFN
4. **SwiGLU** FFN — intermediate_size is typically 8/3 × hidden_size (e.g. 32B: 64 heads × 128 head_dim = 8192, but this is attention, FFN is separate)
5. **RoPE** with standard theta (not specified in paper, but Qwen2.5 uses 1,000,000)
