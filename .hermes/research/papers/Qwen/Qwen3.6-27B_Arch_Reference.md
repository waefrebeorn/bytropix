# Qwen3.6-27B Architecture Reference

**Source:** Hugging Face `config.json`
**HF model:** Qwen/Qwen3.6-27B
**We DON'T have GGUF** for this model locally.

## Overview

- **Type:** Causal Language Model with Vision Encoder
- **Architecture type:** `Qwen3_5ForConditionalGeneration` → `qwen3_5_text` (dense)
- **Params:** 27B total
- **Dtype:** bfloat16
- **Context:** 262,144 native, extensible to ~1M

## Text Model

| Config key | Value |
|------------|-------|
| `hidden_size` | **5120** |
| `intermediate_size` | **17408** (FFN, SwiGLU) |
| `num_hidden_layers` | **64** |
| `num_attention_heads` | **24** (full attention Q heads) |
| `num_key_value_heads` | **4** (full attention KV heads) |
| `head_dim` | **256** |
| `vocab_size` | **248320** |

### Hidden Layout
```
16 × (3 × [Gated DeltaNet → FFN] → 1 × [GQA → FFN])
```
- 64 layers total
- 48 Gated DeltaNet layers (75%)
- 16 GQA full attention layers (25%)

### Gated DeltaNet
| Config key | Value |
|------------|-------|
| `linear_num_key_heads` | **16** |
| `linear_num_value_heads` | **48** |
| `linear_key_head_dim` | **128** |
| `linear_value_head_dim` | **128** |
| `attn_output_gate` | true |
| `output_gate_type` | **swish** (unique to 27B) |

### Full Attention (GQA)
| Config key | Value |
|------------|-------|
| `num_attention_heads` | **24** |
| `num_key_value_heads` | **4** |
| `head_dim` | **256** |

### RoPE
Same as 9B/35B: theta=10M, partial=0.25, MRoPE [11,11,10]

### Vision Encoder
Same architecture as 9B/35B but `out_hidden_size=5120` to match text hidden.

## Note

This is Qwen3.6's **dense flagship** (27B, all active). The 35B-A3B is the MoE variant we actually have locally. The 27B is useful as a reference for what a dense version looks like at larger scale — especially the **swish output gate** which is unique to this model.
