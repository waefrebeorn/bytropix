# Qwen3.6-35B-A3B Architecture Reference

**Source:** Hugging Face `config.json` + model card README
**HF model:** Qwen/Qwen3.6-35B-A3B
**We have GGUF:** Qwen3.6-35B-A3B-UD-IQ2_M (11G) + Ornstein3.6-35B-A3B-SABER-Q2_K (13G)

## Overview

- **Type:** Causal Language Model with Vision Encoder
- **Architecture type:** `Qwen3_5MoeForConditionalGeneration` → `qwen3_5_moe_text`
- **Params:** 35B total, **3B activated** per token
- **Dtype:** bfloat16
- **Context:** 262,144 native, extensible to ~1,010,000 tokens
- This is our **primary target model** for embedding grafting + finetuning

## Text Model

### Dimensions

| Config key | Value |
|------------|-------|
| `hidden_size` | **2048** |
| `moe_intermediate_size` | **512** |
| `num_hidden_layers` | **40** |
| `num_attention_heads` | **16** (full attention Q heads) |
| `num_key_value_heads` | **2** (full attention KV heads) |
| `head_dim` | **256** |
| `vocab_size` | **248320** (padded) |
| `max_position_embeddings` | **262144** |

### Hybrid Attention: Gated DeltaNet + GQA

The hidden layout is:
```
10 × (3 × [Gated DeltaNet → MoE] → 1 × [GQA → MoE])
```

**Every layer ends with MoE** (no plain FFN). The 1 shared expert effectively acts as a skip connection.

#### Gated DeltaNet (linear attention — 75% of layers)
| Config key | Value |
|------------|-------|
| `linear_num_key_heads` | **16** |
| `linear_num_value_heads` | **32** |
| `linear_key_head_dim` | **128** |
| `linear_value_head_dim` | **128** |
| `linear_conv_kernel_dim` | **4** |
| `attn_output_gate` | **true** |

#### GQA / Full Attention (every 4th layer — 25%) — same as dense
| Config key | Value |
|------------|-------|
| `num_attention_heads` | **16** |
| `num_key_value_heads` | **2** |
| `head_dim` | **256** |

**KV heads = 2** for full attention (very aggressive GQA, 8:1 ratio).

### MoE Layer

| Config key | Value |
|------------|-------|
| `num_experts` | **256** (total) |
| `num_experts_per_tok` | **8** (routed) + **1 shared** |
| `moe_intermediate_size` | **512** |
| `shared_expert_intermediate_size` | **512** |
| `router_aux_loss_coef` | 0.001 |
| `output_router_logits` | false |

**NOTE:** Has a **shared expert** — unlike Qwen3 MoE which explicitly removed shared experts. Qwen3.5+ brought them back.

### RoPE

Same as Qwen3.5-9B:
| Config key | Value |
|------------|-------|
| `rope_theta` | **10,000,000** |
| `rope_type` | default |
| `partial_rotary_factor` | **0.25** (64 of 256 dims rotated) |
| `mrope_interleaved` | true |
| `mrope_section` | [11, 11, 10] |

### Normalization & Settings

| Config key | Value |
|------------|-------|
| `rms_norm_eps` | 1e-06 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `tie_word_embeddings` | false |
| `use_cache` | true |
| `hidden_act` | silu |
| `initializer_range` | 0.02 |
| `mamba_ssm_dtype` | float32 |

### MTP

Same as dense: 1 MTP head, no dedicated embeddings.

## Vision Encoder

| Config key | Value |
|------------|-------|
| `model_type` | qwen3_5_moe |
| `depth` | **27** layers |
| `hidden_size` | **1152** |
| `intermediate_size` | **4304** |
| `num_heads` | **16** |
| `patch_size` | **16** |
| `temporal_patch_size` | **2** |
| `spatial_merge_size` | **2** |
| `num_position_embeddings` | **2304** |
| `out_hidden_size` | **2048** (matches text hidden) |
| `hidden_act` | gelu_pytorch_tanh |

## Special Tokens

Same as Qwen3.5-9B: bos=248044, eos=248044, image=248056, video=248057, vs=248053, ve=248054

## Why This Model for WuBu

1. **2048 hidden** — small enough to fit in 6.4GB VRAM, large enough for meaningful representations
2. **Gated DeltaNet** — linear attention is naturally compatible with hyperbolic gyration
3. **3B active** — extremely efficient, leaves room for wubu math overhead
4. **MoE with shared expert** — can be adapted to wubu's expert routing
5. **Vision encoder built-in** — foundation for WuBuVision
