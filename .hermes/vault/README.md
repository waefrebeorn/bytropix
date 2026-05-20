# Research Vault — WuBuText AI (May 19 PM v22)

## Active Papers (Inference-relevant)

| Paper | File | Key Insight |
|-------|------|-------------|
| Qwen3 Technical Report | `qwen-papers/qwen3-technical-report.md` | Architecture: 40 layers, 3:1 SSM/GQA, full_attention_interval=4 |
| DeepSeek-V3 | `deepseek-papers/deepseek-v3-technical-report.md` | MoE: 256 experts, 8 active. MTP: self-speculative decode |
| DeepSeek-V3.2 | `deepseek-papers/deepseek-v3.2-technical-report.md` | DSA: O(L log L) sparse attention, global+local positions |
| DeepSeekMoE | `deepseek-papers/deepseek-moe-architecture.md` | Normalized sigmoid gating, shared experts |
| Qwen2.5-1M | `qwen-papers/qwen2.5-1m-technical-report.md` | Chunked prefill (3-7x), RoPE 4x extrapolation |
| Synthesis | `synthesis.md` | Complete architectural cross-reference (19KB) |

## Quantization References

| Document | Key Info |
|----------|----------|
| `unsloth-qwen3.6-quant-formula.md` | Per-tensor quant types: SSM Q5_K/Q6_K, MoE IQ2_XXS/IQ3_XXS/IQ4_XS, output Q4_K |
| `vault/attention/README.md` | Attention mechanism vault notes |

## DUMP_INTERMEDIATE_DIR Tensor Reference

53 intermediate tensor types per layer from llama.cpp. Key ones for 1:1 debugging:
- `L{N}_conv_input.bin` — SSM conv1d input [32, 2048]
- `L{N}_conv_output_silu.bin` — SSM conv1d output after SiLU [20, 2048]
- `L{N}_Qcur_full.bin` — GQA Q+gate fused projection [5, 8192]
- `L{N}_Kcur.bin` — GQA K projection [5, 512]
- `L{N}_Vcur.bin` — GQA V projection [5, 512]
- `L{N}_linear_attn_out.bin` — SSM recurrence output [5, 2048]
- `L{N}_attn_output.bin` — GQA attention output [5, 2048]
- `L{N}_ffn_moe_out.bin` — MoE output [5, 2048]
- `L{N}_l_out.bin` — Full layer output [5, 2048]

Set `DUMP_INTERMEDIATE_DIR=/tmp/dir` before running `ref_dumper`.
