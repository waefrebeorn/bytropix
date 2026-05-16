# Unsloth Dynamic v2.0 — Qwen3.6-35B-A3B Quantization Formula

## Model
`Qwen3.6-35B-A3B-UD-IQ2_M.gguf` — quantized by Unsloth using Dynamic v2.0

## Architecture (from GGUF header)
- general.architecture = `qwen35moe`
- block_count = 40
- embedding_length = 2048
- SSM inner_size = 4096, state_size = 128, time_step_rank = 32, conv_kernel = 4
- full_attention_interval = 4 (every 4th layer is GQA)
- rope.dimension_sections = [11, 11, 10, 0] (IMRoPE)
- MoE: 256 experts, 8 routed + 1 shared, expert_ff = 512

## Calibration
- `quantize.imatrix.file`: `Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf`
- `quantize.imatrix.dataset`: `unsloth_calibration_Qwen3.6-35B-A3B.txt` (custom chat-calibration, not wikitext)
- `quantize.imatrix.entries_count`: 510
- `quantize.imatrix.chunks_count`: 76

## Per-Tensor Quantization (the actual formula)

| Tensor | Shape | Quant | bpw | Count |
|--------|-------|-------|-----|-------|
| **SSM bottleneck** | | | | |
| `ssm_a` | [32] | F32 | 32.0 | 30 |
| `ssm_dt.bias` | [32] | F32 | 32.0 | 30 |
| `ssm_norm.weight` | [128] | F32 | 32.0 | 30 |
| `ssm_beta.weight` | [2048, 32] | F32 | 32.0 | 30 |
| `ssm_alpha.weight` | [2048, 32] | F32 | 32.0 | 30 |
| `ssm_conv1d.weight` | [4, 8192] | F32 | 32.0 | 30 |
| **SSM projections** | | | | |
| `attn_qkv.weight` | [2048, 8192] | Q5_K | 6.5 | 30 |
| `attn_gate.weight` | [2048, 4096] | Q5_K | 6.5 | 30 |
| `ssm_out.weight` | [4096, 2048] | Q6_K | 7.5 | 30 |
| **GQA attention** | | | | |
| `attn_q.weight` | [2048, 8192] | Q5_K | 6.5 | 10 |
| `attn_k.weight` | [2048, 512] | Q5_K | 6.5 | 10 |
| `attn_v.weight` | [2048, 512] | Q5_K | 6.5 | 10 |
| `attn_output.weight` | [4096, 2048] | Q5_K | 6.5 | 10 |
| **MoE routed experts** | | | | |
| `ffn_down_exps.weight` | [512, 2048, 256] | IQ3_XXS | ~3.3 | 37 |
| `ffn_down_exps.weight` | [512, 2048, 256] | IQ4_XS | ~4.3 | 3 |
| `ffn_gate_exps.weight` | [2048, 512, 256] | IQ2_XXS | ~2.2 | 40 |
| `ffn_up_exps.weight` | [2048, 512, 256] | IQ2_XXS | ~2.2 | 40 |
| **MoE shared expert** | | | | |
| `ffn_down_shexp.weight` | [512, 2048] | Q6_K | 7.5 | 40 |
| `ffn_gate_shexp.weight` | [2048, 512] | Q5_K | 6.5 | 40 |
| `ffn_up_shexp.weight` | [2048, 512] | Q5_K | 6.5 | 40 |
| **MoE router** | | | | |
| `ffn_gate_inp.weight` | [2048, 256] | F32 | 32.0 | 40 |
| `ffn_gate_inp_shexp.weight` | [2048] | F32 | 32.0 | 40 |
| **Norms** | | | | |
| `attn_norm.weight` | [2048] | F32 | 32.0 | 40 |
| `post_attention_norm.weight` | [2048] | F32 | 32.0 | 40 |
| **Embedding / Output** | | | | |
| `token_embd.weight` | [2048, 248320] | Q5_K | 6.5 | 1 |
| `output.weight` | [2048, 248320] | Q4_K | 5.0 | 1 |
| `output_norm.weight` | [2048] | F32 | 32.0 | 1 |

## Key Principles

1. **SSM bottleneck tensors kept in F32** — ssm_a, ssm_dt_bias, ssm_norm, ssm_beta, ssm_alpha, ssm_conv1d all stay at full precision. These are small tensors but critical for SSM recurrence correctness.

2. **MoE experts get heaviest compression** — IQ2_XXS (~2.2 bpw) for gate/up, IQ3_XXS (~3.3 bpw) for down. 37/40 layers use IQ3_XXS, 3 layers use IQ4_XS (slightly higher precision for those layers).

3. **SSM output projection at Q6_K** — higher precision than SSM QKV (Q5_K), because the output projection determines the layer's contribution to the residual stream.

4. **Shared expert at Q5_K/Q6_K** — not compressed as heavily as routed experts, preserving shared knowledge.

5. **MoE router at F32** — critical for correct expert selection; any router error compounds through MoE.

6. **GQA at uniform Q5_K** — standard attention gets consistent treatment.

## Necessary Dequant Support
For our GGUF reader to load this model:
- F32 ✓, F16 ✓
- Q4_K ✓, Q5_K ✓, Q6_K ✓, Q8_0 ✓
- IQ2_XXS ✓, IQ2_S ✓
- IQ3_XXS ✓, IQ3_S ✓
- IQ1_S ✓, IQ1_M ❌ (enum wrong: 23→29)
- IQ4_NL ❌ (missing: need enum=20 + dequant)
- IQ4_XS ❌ (missing: need enum=23 + dequant)
