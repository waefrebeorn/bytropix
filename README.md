# wubuwizard

C11 inference engine ("the Colonel") plus a vault of mathematical encoders and
prior research. No external ML dependencies; quantization, matmul, tokenizer,
and model loaders are implemented in-tree.

## Layout

| Path | Contents |
|------|----------|
| `src/` | Core engine: model loader, SSM/GQA/MoE forward, quant matmul, GGUF + safetensors readers, CUDA kernels. |
| `include/` | Public headers (opaque structs, minimal includes). |
| `tools/` | `gen_text` (CPU generation), verification harnesses, component tests, API server, analysis scripts. |
| `vault/` | Quantization format references, legacy docs, archived session snapshots. |
| `THEORY/` | Research papers (markdown), `math_viz/` runnable proofs, `ENCODERS/` converter notes. |
| `draftPY/` | Python research prototypes (hyperbolic/GAAD/DFT/DCT encoders). |

## Build

```bash
make gen_text          # CPU inference binary
make test_ssd_moe      # slot-bank unit test (synthetic)
make test_real_load    # loads real Agents-A1-4B shards
make api_server        # OpenAI-compatible HTTP server
```

C11, GCC/Clang. CUDA paths require `nvcc` (optional). Flags in `Makefile`.

## Model support

Loads GGUF and HuggingFace safetensors. Tensor-name mapping + dimension
derivation live in `src/wubu_model_safetensors_bridge.c`. Per-model status is in
[STATUS.md](STATUS.md). Supported architectures include Gated-DeltaNet hybrid
(SSM + GQA + MLP), Qwen3.5 MoE (256 experts), dense 27B, and LoRA adapters.

## Subsystems

- **SSM forward** (`wubu_ssm.c`): Gated DeltaNet recurrence, conv1d, GQA, QK-norm.
- **MoE** (`wubu_moe.c`): router, shared expert, quantized experts; plus an
  SSD-paged variant (`wubu_moe_forward_ssd`) backed by the slot-bank.
- **ds4-ssd slot-bank** (`wubu_ssd_moe.c`): dense tensors resident in RAM; routed
  MoE experts paged from an SSD sidecar via a per-layer LRU slot-bank. See
  [docs/ssd_moe.md](docs/ssd_moe.md).
- **Tokenizers**: GPT-2 BPE (`wubu_tokenizer.c`) and HF BPE (`wubu_tokenizer_hf.c`).
- **Quantization**: Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ3_XXS/IQ4_XS + Q8_0 activations,
  self-hosted `vec_dot` (no libggml). Q4_0 KV cache.

## Vault

The `THEORY/` and `vault/` trees hold the underlying math: Poincaré/hyperbolic
embeddings, GAAD (Golden Aspect Adaptive Decomposition), DFT/DCT spectral
encoders, the tailslayer hedged-read system, and the DeepSeek MoE/sparse-attention
paper set. `THEORY/math_viz/` contains runnable numerical proofs.
