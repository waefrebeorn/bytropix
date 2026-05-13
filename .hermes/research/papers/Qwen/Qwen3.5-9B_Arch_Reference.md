# Qwen3.5-9B Architecture Reference

**Source:** Hugging Face `config.json` + model card README
**HF model:** Qwen/Qwen3.5-9B
**We have GGUF:** Qwen3.5-9B-Q4_K_M (5.3G) + Qwen3.5-9B-UD-IQ2_M (3.4G)

## Overview

- **Type:** Causal Language Model with Vision Encoder
- **Architecture type:** `Qwen3_5ForConditionalGeneration` → `qwen3_5_text`
- **Params:** 9B total
- **Dtype:** bfloat16
- **Context:** 262,144 native, extensible to ~1,010,000 tokens

## Text Model

### Dimensions

| Config key | Value |
|------------|-------|
| `hidden_size` | **4096** |
| `intermediate_size` | **12288** (FFN, SwiGLU) |
| `num_hidden_layers` | **32** |
| `num_attention_heads` | **16** (full attention Q heads) |
| `num_key_value_heads` | **4** (full attention KV heads) |
| `head_dim` | **256** |
| `vocab_size` | **248320** (padded) |
| `max_position_embeddings` | **262144** |

### Hybrid Attention: Gated DeltaNet + GQA

The hidden layout is:
```
8 × (3 × [Gated DeltaNet → FFN] → 1 × [GQA → FFN])
```

#### Gated DeltaNet (linear attention — 75% of layers)
| Config key | Value |
|------------|-------|
| `linear_num_key_heads` | **16** |
| `linear_num_value_heads` | **32** |
| `linear_key_head_dim` | **128** |
| `linear_value_head_dim` | **128** |
| `linear_conv_kernel_dim` | **4** |
| `attn_output_gate` | **true** |

**Every layer** has linear attention (DeltaNet) except full attention every 4th layer.

#### GQA / Full Attention (every 4th layer — 25%)
| Config key | Value |
|------------|-------|
| `num_attention_heads` | **16** |
| `num_key_value_heads` | **4** |
| `head_dim` | **256** |

### RoPE

| Config key | Value |
|------------|-------|
| `rope_theta` | **10,000,000** |
| `rope_type` | default |
| `partial_rotary_factor` | **0.25** (64 of 256 dims are rotated) |
| `mrope_interleaved` | true |
| `mrope_section` | [11, 11, 10] (32 total dims, MRoPE) |

MRoPE = multi-resolution RoPE for 3D spatiotemporal position encoding (used by vision).

### Normalization

| Config key | Value |
|------------|-------|
| `rms_norm_eps` | 1e-06 |
| `attention_bias` | **false** (no QKV bias) |
| `attention_dropout` | 0.0 |

### MTP (Multi-Token Prediction)

| Config key | Value |
|------------|-------|
| `mtp_num_hidden_layers` | 1 |
| `mtp_use_dedicated_embeddings` | false |

### Other Settings

| Config key | Value |
|------------|-------|
| `tie_word_embeddings` | **false** (no weight tying) |
| `use_cache` | true |
| `hidden_act` | silu |
| `initializer_range` | 0.02 |

## Vision Encoder

| Config key | Value |
|------------|-------|
| `model_type` | qwen3_5 |
| `depth` | **27** layers |
| `hidden_size` | **1152** |
| `intermediate_size` | **4304** |
| `num_heads` | **16** |
| `patch_size` | **16** |
| `temporal_patch_size` | **2** |
| `spatial_merge_size` | **2** |
| `num_position_embeddings` | **2304** |
| `out_hidden_size` | **4096** (matches text hidden) |
| `in_channels` | 3 |
| `hidden_act` | gelu_pytorch_tanh |
| `deepstack_visual_indexes` | [] (empty) |

## Special Tokens

| Token | ID |
|-------|----|
| bos | 248044 |
| eos | 248044 |
| pad | null |
| image_token | 248056 |
| video_token | 248057 |
| vision_start | 248053 |
| vision_end | 248054 |
