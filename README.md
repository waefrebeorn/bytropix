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
[STATUS.md](STATUS.md).

### Supported architectures

| Architecture | What it is | Key files |
|---|---|---|
| **Gated-DeltaNet hybrid** (SSM + GQA + MLP) | Qwen3.x / Agents-A1; per-layer `layer_types` selects SSM vs GQA | `src/wubu_ssm.c`, `src/wubu_model_safetensors_bridge.c` |
| **Qwen3.x MoE** (256 experts) | Deep MoE with shared expert + routed experts; SSD-paged | `src/wubu_moe.c`, `src/wubu_ssd_moe.c` |
| **Dense hybrid** (Qwen3.6-27B) | 64-layer dense, SSM+GQA per layer | `src/wubu_ssm.c` |
| **LoRA adapters** | BTL-3 rank-32 alpha-64 on Qwen3.6-27B base | `src/wubu_lora.c` |

### Model status matrix

| Model | HF repo | D | Layers | Experts | Key |
|---|---|---|---|---|---|
| **Agents-A1-4B** | `InternScience/agents-a1` | 2560 | 32 | 0 (dense) | Shards present; real SSM forward verified ✅ |
| **KAT-Coder-V2.5-Dev** | `Kwaipilot/KAT-Coder-V2.5-Dev` | 2048 | 40 | 256 | 13/13 shards present; SSD slot-bank working |
| **Qwen3.6-27B** | `Qwen/Qwen3.6-27B` | 5120 | 64 | 0 (dense hybrid) | Shards partial; real SSM forward finite ✅ |
| **BTL-3** | `badtheorylabs/BTL-3` | — | — | — | LoRA on Qwen3.6-27B; adapter downloading |

The bridge also handles legacy GGUF checkpoints (see `src/wubu_model.c`).
All tensor loading goes through `wubu_shard_open` + `wubu_model_safetensors_bridge.c`;
the lazy BF16 path (mmap + on-demand dequant) keeps 27B-class models under 13 GB.

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
