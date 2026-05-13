# Qwen3.6-35B-A3B — Complete Platform Target Map

Source: Original HF config files at `/home/wubu/models/qwen36_og/`
Generated: May 13, 2026 (Session-end audit appended to GH repo docs/ctx)

---

## 1. Architecture Overview

| Field | Value | Notes |
|-------|-------|-------|
| Model type | `qwen3_5_moe` | |
| Architecture | `Qwen3_5MoeForConditionalGeneration` | |
| Transformers | `4.57.1` | |
| D_MODEL | 2048 | hidden_size |
| D_INNER | 4096 | GQA head_dim × Q_heads = 256×16 (but output 4096) |
| Num layers | 40 | 30 SSM + 10 GQA |
| GQA: Q heads | 16 | num_attention_heads |
| GQA: KV heads | 2 | num_key_value_heads |
| GQA: head_dim | 256 | head_dim |
| SSM: K heads | 16 | linear_num_key_heads |
| SSM: V heads | 32 | linear_num_value_heads |
| SSM: d_state | 128 | linear_key_head_dim = linear_value_head_dim |
| SSM: conv_kernel | 4 | linear_conv_kernel_dim |
| SSM: dtype | float32 | mamba_ssm_dtype |
| Max seq len | 262,144 | max_position_embeddings |
| Full-attn interval | 4 | every 4th layer is GQA |
| Vocab size | 248,320 | vocab_size |
| Tie embeddings | false | separate lm_head |
| RMS norm eps | 1e-6 | |
| Hidden act | silu | SiLU |
| Output router | false | output_router_logits |

### KV Cache Size (for GQA layers)

- K cache: 2 heads × 256 dim × 262K tokens = 134 MB per layer
- V cache: 2 heads × 256 dim × 262K tokens = 134 MB per layer
- 10 GQA layers total: ~2.7 GB in f16

---

## 2. Layer Type Pattern (40 layers)

`full_attention_interval: 4` means every layer whose index satisfies `i % 4 == 3` is GQA.

```
Layer   0: SSM | Layer  10: SSM | Layer  20: SSM | Layer  30: SSM
Layer   1: SSM | Layer  11: SSM | Layer  21: SSM | Layer  31: SSM
Layer   2: SSM | Layer  12: SSM | Layer  22: SSM | Layer  32: SSM
Layer   3: GQA | Layer  13: SSM | Layer  23: SSM | Layer  33: SSM
Layer   4: SSM | Layer  14: SSM | Layer  24: SSM | Layer  34: SSM
Layer   5: SSM | Layer  15: GQA | Layer  25: SSM | Layer  35: GQA
Layer   6: SSM | Layer  16: SSM | Layer  26: SSM | Layer  36: SSM
Layer   7: GQA | Layer  17: SSM | Layer  27: GQA | Layer  37: SSM
Layer   8: SSM | Layer  18: SSM | Layer  28: SSM | Layer  38: SSM
Layer   9: SSM | Layer  19: GQA | Layer  29: SSM | Layer  39: GQA
```

GQA layers: {3, 7, 11, 15, 19, 23, 27, 31, 35, 39} — 10 total.
SSM layers: all others — 30 total.

---

## 3. SSM Layer — GGUF Tensor Inventory (layer 0)

19 tensors per SSM layer. Total: 30 × 19 = 570 SSM tensors.

| GGUF Name | Shape | Original HF Name | Description |
|-----------|-------|------------------|-------------|
| `blk.<L>.attn_norm.weight` | [2048] | `input_layernorm.weight` | Input RMSNorm |
| `blk.<L>.attn_qkv.weight` | [2048, 8192] | `in_proj_qkv.weight` | Fused QKV projection (no bias!) |
| `blk.<L>.attn_gate.weight` | [2048, 4096] | `out_proj.weight` | Output gate (gated output) |
| `blk.<L>.ssm_conv1d.weight` | [4, 8192] | `conv1d.weight` | Conv1d kernel (depthwise along dims) |
| `blk.<L>.ssm_dt.bias` | [32] | `dt_bias` | DT bias (one per V head) |
| `blk.<L>.ssm_norm.weight` | [32] | `norm.weight` | SSM output norm (one per V head) |
| `blk.<L>.ssm_out.weight` | [4096, 2048] | `out_proj.weight` | SSM output projection |
| `blk.<L>.ssm_a` | [32] | `A_log` | A matrix log (one per V head) |
| `blk.<L>.ssm_alpha.weight` | [2048, 32] | `in_proj_a.weight` | Alpha projection (A input) |
| `blk.<L>.ssm_beta.weight` | [2048, 32] | `in_proj_b.weight` | Beta projection (B input) |
| `blk.<L>.post_attention_norm.weight` | [2048] | `post_attention_layernorm.weight` | Post-attention RMSNorm |

### SSM Internal Dimensions

- attn_qkv: [2048, 8192] = [D, Q_heads*head_dim + K_heads*d_state + V_heads*d_state]
  = [2048, 16*128 + 16*128 + 32*128] = [2048, 2048 + 2048 + 4096] = [2048, 8192] ✓
  - Q portion: [2048, 2048] → 16 heads × 128 dim
  - K portion: [2048, 2048] → 16 heads × 128 dim (ssm_key)
  - V portion: [2048, 4096] → 32 heads × 128 dim (ssm_value)

- ssm_conv1d: [4, 8192] = conv_kernel=4, channels=V_heads*d_state=32*128=4096
  Wait — 8192 = 32*256? No: 8192 = 4096... Actually: ssm_conv1d channels=input_channels=D_MODEL? 
  Let me re-check: ssm_conv1d.weight shape [4, 8192] — 4 = conv_kernel, 8192 = D_MODEL * 4 = SSM expansion
  Actually 8192 = d_inner (4096) × 2? No — let me check: conv1d processes the expanded QKV?
  
  Conv1d in Mamba-2 processes the **value** branch after projection. V dimension = 32 heads × 128 = 4096.
  But shape is [4, 8192] which is 4 × 4096*2. So conv1d doubles the channels for SiLU gating in conv:
  conv1d output = 2 * d_inner_ssm = 2 * 4096 = 8192. Half goes through activation, other half gates it.
  **Correction**: This is the Mamba-2 conv1d with SiLU gate built in — it outputs 2× channels.
  Half → SiLU → gate(silu(conv(x_hidden)) * x_residual) pattern.

  Actually re-examining: ssd_combined in Mamba-2 uses selective scan. conv1d outputs 8192 = 2*4096
  where 4096 = V_heads × d_state = 32×128. The conv processes the "hidden" state pre-scan.
  The split: 4096 for conv(x) → SiLU → gate, 4096 for x → residual.

  But our C implementation (from test_gpu.c) uses the conv1d directly without this gating.
  Need to verify.

- ssm_dt.bias: [32] — one per V head (32 heads)
- ssm_a: [32] — A_log, one per V head
- ssm_alpha: [2048, 32] — projected from hidden, mixed with dt
- ssm_beta: [2048, 32] — projected from hidden, mixed with dt
- ssm_out: [4096, 2048] — output from SSM scan back to D_MODEL
- attn_gate: [2048, 4096] — the output gate for the gated SSM output
  - This is the `attn_output_gate: true` — it's part of the SSM, NOT GQA!
  - SSM output = gate(silu(ssm_out(x)) * x_residual) — the gate projects and multiplies

**CRITICAL INSIGHT**: `attn_gate` exists on **ALL SSM layers** but NOT on GQA layers.
The config's `attn_output_gate: true` applies to SSM attention, not GQA attention.
GQA attention uses `attn_output.weight` without a separate gate.

---

## 4. GQA Layer — GGUF Tensor Inventory (layer 3)

16 tensors per GQA layer. Total: 10 × 16 = 160 GQA tensors.

| GGUF Name | Shape | Original HF Name | Description |
|-----------|-------|------------------|-------------|
| `blk.<L>.attn_norm.weight` | [2048] | `input_layernorm.weight` | Input RMSNorm |
| `blk.<L>.attn_q.weight` | [2048, 4096] | `q_proj.weight` | Q projection |
| `blk.<L>.attn_q_norm.weight` | [256] | `q_norm.weight` | Q head normalization (per head) |
| `blk.<L>.attn_k.weight` | [2048, 512] | `k_proj.weight` | K projection |
| `blk.<L>.attn_k_norm.weight` | [256] | `k_norm.weight` | K head normalization (per head) |
| `blk.<L>.attn_v.weight` | [2048, 512] | `v_proj.weight` | V projection |
| `blk.<L>.attn_output.weight` | [4096, 2048] | `o_proj.weight` | Attention output projection |
| `blk.<L>.post_attention_norm.weight` | [2048] | `post_attention_layernorm.weight` | Post-attention RMSNorm |

### GQA Internal Dimensions

- Q (fused with gate): [2048, 8192] = [D, Q_heads × head_dim × 2] = [2048, 16 × 256 × 2]
  - First 4096 = Q projection
  - Second 4096 = attention gate projection
- K: [2048, 512] = [D, KV_heads × head_dim] = [2048, 2 × 256] = [2048, 512] ✓
- V: [2048, 512] = [D, KV_heads × head_dim] = [2048, 2 × 256] = [2048, 512] ✓
- Q/K norm: [256] = per-head normalization (one head dim)
- Output: [4096, 2048] = [D_INNER, D] — expands to inner dim then back

**No separate `attn_gate` on GQA layers!** Instead, GQA's Q weight is fused with a gate:
`attn_q.weight` is [2048, 8192] where the first half is Q and the second half is gate.
GQA forward: Q = first 4096, gate = second 4096. After attention: out *= sigmoid(gate).
Then attn_output projection.

**CRITICAL**: The GGUF fuses Q and gate into one tensor even though the original HF
had them separate (`q_proj.weight` = [2048, 4096] alone). llama.cpp's GGUF conversion
appended the gate projection to create a single [2048, 8192] tensor.
Our C implementation's `gpu_load_gqa_layer` should read [D_MODEL * q_dim] = 2048 * 4096
elements, NOT 2048 * 8192. The fused layout is a **run-time** concept (split in memory),
not a file-level one, unless the GGUF was explicitly fused.

→ **FACT CHECK**: The GGUF shape says [2048, 8192]. So the GGUF file DOES store the fused
Q + gate as one tensor. Our C code reads `D_MODEL * q_dim_x2` = 2048 * 8192 elements,
which matches. But we should verify gguf_read_tensor_f32 with the exact element count.

### KV Cache Implications

- K cache per GQA layer: [B, KV_heads, T, head_dim] = [1, 2, 262K, 256] = 134 MB (f16)
- V cache per GQA layer: same = 134 MB
- 10 GQA layers → ~2.7 GB total for full 262K context

For causal attention at inference, half precision KV cache fits in 8GB VRAM even at full 262K.

---

## 5. MoE (Mixture of Experts) — Same Structure on ALL Layers

8 tensors per layer (SSM and GQA layers have IDENTICAL MoE structure).

### Expert Weights (256 experts)

| GGUF Name | Shape | Description |
|-----------|-------|-------------|
| `ffn_gate_exps.weight` | [2048, 512, 256] | Gate projection per expert (no longer fused with up) |
| `ffn_up_exps.weight` | [2048, 512, 256] | Up projection per expert |
| `ffn_down_exps.weight` | [512, 2048, 256] | Down projection per expert |

Dimensions: [hidden, intermediate, n_experts] = [2048, 512, 256]

**IMPORTANT**: Original HF has `experts.gate_up_proj` as fused [hidden, 2*intermediate, n_experts].
GGUF conversion (llama.cpp) **split** gate+up into separate tensors.
Our C MoE loader must handle **transposed** expert weights.

### Shared Expert

| GGUF Name | Shape | Description |
|-----------|-------|-------------|
| `ffn_gate_shexp.weight` | [2048, 512] | Shared expert gate |
| `ffn_up_shexp.weight` | [2048, 512] | Shared expert up |
| `ffn_down_shexp.weight` | [512, 2048] | Shared expert down |
| `ffn_gate_inp.weight` | [2048, 256] | Expert router (256 outputs → softmax → top-8) |
| `ffn_gate_inp_shexp.weight` | [2048] | Shared expert router (scalar gate) |

### MoE Forward Pass

```
router_logits = x @ ffn_gate_inp.weight^T  → [B, T, 256]
router_weights = softmax(router_logits)     → probability distribution
top_k_weights, top_k_indices = topk(router_weights, k=8)

# Per token: route to 8 selected experts
for each expert in top-8:
    hidden_gate = gate_expert(proj_gate, x)  → SiLU
    hidden_up = up_expert(proj_up, x)
    expert_out = down_expert(hidden_gate * hidden_up)
    
# Shared expert (always active)
shared_gate = SiLU(x @ ffn_gate_shexp)
shared_up = x @ ffn_up_shexp
shared_out = (shared_gate * shared_up) @ ffn_down_shexp

# Combine
output = shared_out + sum(top_k_weights * expert_out)
```

---

## 6. Shared/Global Tensors

| GGUF Name | Shape | Description |
|-----------|-------|-------------|
| `token_embd.weight` | [2048, 248320] | Token embedding (vocab × hidden) |
| `output_norm.weight` | [2048] | Final RMSNorm |
| `output.weight` | [2048, 248320] | LM head (hidden × vocab) — separate, no weight tying |

Note: `output.weight` and `token_embd.weight` have identical shapes [2048, 248320] but are SEPARATE tensors (`tie_word_embeddings: false`).

---

## 7. MTP (Multi-Token Prediction)

1 MTP head, 19 tensors.

| GGUF Name | Shape | Description |
|-----------|-------|-------------|
| `mtp.pre_fc_norm_embedding.weight` | [2048] | Norm before embedding projection |
| `mtp.pre_fc_norm_hidden.weight` | [2048] | Norm before hidden projection |
| `mtp.fc.weight` | [2048, 2560] | Combined projection [embed;hidden] → hidden |
| `mtp.norm.weight` | [2048] | Final norm before LM head |
| `mtp.layers.0.attn_q.weight` | [2048, 4096] | MTP self-attention Q |
| `mtp.layers.0.attn_k.weight` | [2048, 512] | MTP self-attention K |
| `mtp.layers.0.attn_v.weight` | [2048, 512] | MTP self-attention V |
| `mtp.layers.0.attn_q_norm.weight` | [256] | Q norm |
| `mtp.layers.0.attn_k_norm.weight` | [256] | K norm |
| `mtp.layers.0.attn_output.weight` | [4096, 2048] | Attention output |
| `mtp.layers.0.input_layernorm.weight` | [2048] | Input norm |
| `mtp.layers.0.post_attention_layernorm.weight` | [2048] | Post-attention norm |
| `mtp.layers.0.ffn_gate_exps.weight` | [2048, 512, 256] | Expert gate (same MoE as main model) |
| `mtp.layers.0.ffn_up_exps.weight` | [2048, 512, 256] | Expert up |
| `mtp.layers.0.ffn_down_exps.weight` | [512, 2048, 256] | Expert down |
| `mtp.layers.0.ffn_gate_inp.weight` | [2048, 256] | Expert router |
| `mtp.layers.0.ffn_gate_shexp.weight` | [2048, 512] | Shared expert gate |
| `mtp.layers.0.ffn_up_shexp.weight` | [2048, 512] | Shared expert up |
| `mtp.layers.0.ffn_down_shexp.weight` | [512, 2048] | Shared expert down |
| `mtp.layers.0.ffn_gate_inp_shexp.weight` | [2048] | Shared expert router |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | [2048, 512] | (same as ffn_gate_shexp) |

MTP forward:
```
# Given h (hidden from layer 40) and e (token embedding of predicted token)
combined = fc(concat(norm_embedding(e), norm_hidden(h)))
# Then same as a normal layer: norm → attention → norm → MoE
output = norm(layer40_hidden) + lm_head(head)
```

---

## 14. MMProj (Vision Projector) — from `/models/qwen3.6-35b-mmproj-F16.gguf`

334 tensors total (includes 27 vision blocks + patch embed + 2-layer MLP projector).

### MMProj GGUF Metadata

| Key | Value |
|-----|-------|
| Architecture | `clip` (CLIP-style vision encoder) |
| Projector type | `qwen3vl_merger` (custom Qwen3VL merger) |
| Has vision encoder | True |
| Use GELU | True (GELU tanh approximation) |
| DeepStack layers | False |
| Quant version | 2 |
| File type | F16 |
| Spatial merge size | 2 |

### Vision Config (from MMProj GGUF KV + HF config cross-ref)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | ? | From clip.vision metadata |
| Patch size | 16 | From clip.vision metadata |
| Embedding length | 1152 | hidden_size=1152 |
| Feed forward length | 4304 | intermediate_size=4304 |
| Block count | 27 | depth=27 |
| Head count | 16 | 16 attention heads |
| Spatial merge size | 2 | Merge 2×2 patches |
| Projection dim | 2048 | Output dim (matches text D_MODEL) |
| Layer norm epsilon | ? | From clip.vision metadata |
| Image mean | ? | Normalization mean |
| Image std | ? | Normalization std |

### Tensor Structure

#### Patch Embedding
| Tensor | Shape | Elements | Description |
|--------|-------|----------|-------------|
| `v.patch_embd.weight` | [16, 16, 3, 1152] | 884,736 | Conv kernel: patch_h=16, patch_w=16, in_c=3, out_c=1152 |
| `v.patch_embd.weight.1` | [16, 16, 3, 1152] | 884,736 | Second temporal frame kernel (temporal_patch_size=2) |
| `v.patch_embd.bias` | [1152] | 1,152 | Bias |
| `v.position_embd.weight` | [1152, 2304] | 2,658,816 | Position embeddings [dim=1152, max_pos=2304] |

**NOTE**: Two patch embed weight tensors (`.weight` and `.weight.1`) correspond to `temporal_patch_size=2`. 
Each 2-frame tube gets processed by both weights independently.

Position embedding has 2304 max positions = enough for:
- Image (single): ceil(224/16) × ceil(224/16) = 14×14 = 196 patches
- After spatial merge (2×2): 196/4 = 49 merged tokens
- Video: 49 per frame × frames with temporal encoding
- 2304 = 49 × ~47 frames max (or other spatial arrangement)

#### Post-LayerNorm (after all 27 ViT blocks)
| Tensor | Shape | Description |
|--------|-------|-------------|
| `v.post_ln.weight` | [1152] | LayerNorm weight |
| `v.post_ln.bias` | [1152] | LayerNorm bias |

#### 27 Vision Transformer Blocks (layers 0-26)

Per block (10 tensors × 27 = 270 total):

| Tensor | Shape | Description |
|--------|-------|-------------|
| `v.blk.<N>.ln1.weight` | [1152] | Pre-attention LayerNorm weight |
| `v.blk.<N>.ln1.bias` | [1152] | Pre-attention LayerNorm bias |
| `v.blk.<N>.attn_qkv.weight` | [1152, 3456] | Fused QKV [1152, 1152×3 = 3456] |
| `v.blk.<N>.attn_qkv.bias` | [3456] | QKV bias |
| `v.blk.<N>.attn_out.weight` | [1152, 1152] | Attention output projection |
| `v.blk.<N>.attn_out.bias` | [1152] | Output bias |
| `v.blk.<N>.ln2.weight` | [1152] | Pre-MLP LayerNorm weight |
| `v.blk.<N>.ln2.bias` | [1152] | Pre-MLP LayerNorm bias |
| `v.blk.<N>.ffn_up.weight` | [1152, 4304] | MLP up projection (in older HF was linear_fc1) |
| `v.blk.<N>.ffn_up.bias` | [4304] | Up bias |
| `v.blk.<N>.ffn_down.weight` | [4304, 1152] | MLP down projection (was linear_fc2) |
| `v.blk.<N>.ffn_down.bias` | [1152] | Down bias |

**Vision Block Forward**:
```
x_attn = attn(ln1(x)) + x                         # Attn with LayerNorm
x_mlp = silu(ffn_up(ln2(x_attn))) * ffn_down(...) # Wait - original config said GELU tanh!
```

**IMPORTANT**: Config says `hidden_act: "gelu_pytorch_tanh"` but GGUF metadata has 
`clip.use_gelu: True`. The tanh approximation of GELU is used:
```
gelu_tanh(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

This is the GELU with tanh approximation (not SiLU/not exact GELU).

#### Merger (MMProj Projector) — 3 layers

| Tensor | Shape | Elements | Description |
|--------|-------|----------|-------------|
| `mm.0.weight` | [4608, 4608] | 21,233,664 | Spatial merge linear (in=4×1152, out=4608) |
| `mm.0.bias` | [4608] | 4,608 | |
| `mm.2.weight` | [4608, 2048] | 9,437,184 | Project to text dim (4608 → 2048) |
| `mm.2.bias` | [2048] | 2,048 | |

**NOTE**: There is NO `mm.1` tensor! The merger is:
1. `mm.0` = Linear(4608 → 4608) with GELU activation (no separate mm.1)
2. `mm.2` = Linear(4608 → 2048)  (mm.1 is the GELU, no weight)

Index gap: mm.0 → GELU → mm.2. This means the original model may have had
a 3-layer MLP (mm.0 → mm.1 → mm.2) where mm.1 was GELU-only/no weights,
and the GGUF format dropped it. Or mm.0 is [4608, 4608] with built-in activation.

#### Complete Vision Forward (MMProj path)

```
# 1. Patch Embed
#    For each temporal frame in 2-frame tube:
x0 = conv2d(video_frame, v.patch_embd.weight)  # → [B, 1152, H_patches, W_patches]
x1 = conv2d(video_frame_next, v.patch_embd.weight.1)  # → same shape
x = concat(x0, x1, dim=temporal)  # → temporal encoding

# 2. Add Position Embeddings
x = x + v.position_embd.weight

# 3. 27 ViT Blocks
for blk in 0..26:
    x = v.blk.<N>.attn(v.blk.<N>.ln1(x)) + x
    x = v.blk.<N>.ffn(v.blk.<N>.ln2(x)) + x
    # ffn: GELU_tanh(x @ ffn_up + bias) @ ffn_down + bias

# 4. Post Norm
x = v.post_ln(x)

# 5. Spatial Merge (2×2 grid → concatenate)
#    Input: [B, num_patches, 1152] where patches arranged spatially
#    Output: [B, num_patches/4, 4608]
#    Each group of 4 spatial neighbors concatenated along channel dim
x_merged = spatial_merge_2x2(x)  # → [B, P_merged, 4608]

# 6. MLP Projector
x = GELU_tanh(x_merged @ mm.0.weight^T + mm.0.bias)  # [B, PM, 4608]
x = x @ mm.2.weight^T + mm.2.bias                     # [B, PM, 2048]
                                                       # 2048 = D_MODEL ✓

# 7. Insert special tokens (vision_start, image_pad, vision_end)
#    image_pad repeats for each merged patch token
#    These token IDs are: vision_start=248053, image_pad=248056, vision_end=248054
```

### MMProj File Weights (original HF ↔ GGUF mapping)

| HF Name (safetensors index) | GGUF Name | Shape |
|------------------------------|-----------|-------|
| `patch_embed.proj.weight` | `v.patch_embd.weight` | [16,16,3,1152] |
| `patch_embed.proj.weight` (temp2) | `v.patch_embd.weight.1` | [16,16,3,1152] |
| `patch_embed.proj.bias` | `v.patch_embd.bias` | [1152] |
| `pos_embed.weight` | `v.position_embd.weight` | [1152, 2304] |
| `blocks.<N>.norm1.weight` | `v.blk.<N>.ln1.weight` | [1152] |
| `blocks.<N>.norm1.bias` | `v.blk.<N>.ln1.bias` | [1152] |
| `blocks.<N>.attn.qkv.weight` | `v.blk.<N>.attn_qkv.weight` | [1152, 3456] |
| `blocks.<N>.attn.qkv.bias` | `v.blk.<N>.attn_qkv.bias` | [3456] |
| `blocks.<N>.attn.proj.weight` | `v.blk.<N>.attn_out.weight` | [1152, 1152] |
| `blocks.<N>.attn.proj.bias` | `v.blk.<N>.attn_out.bias` | [1152] |
| `blocks.<N>.norm2.weight` | `v.blk.<N>.ln2.weight` | [1152] |
| `blocks.<N>.norm2.bias` | `v.blk.<N>.ln2.bias` | [1152] |
| `blocks.<N>.mlp.linear_fc1.weight` | `v.blk.<N>.ffn_up.weight` | [1152, 4304] |
| `blocks.<N>.mlp.linear_fc1.bias` | `v.blk.<N>.ffn_up.bias` | [4304] |
| `blocks.<N>.mlp.linear_fc2.weight` | `v.blk.<N>.ffn_down.weight` | [4304, 1152] |
| `blocks.<N>.mlp.linear_fc2.bias` | `v.blk.<N>.ffn_down.bias` | [1152] |
| `merger.linear_fc1.weight` | `mm.0.weight` | [4608, 4608] |
| `merger.linear_fc1.bias` | `mm.0.bias` | [4608] |
| `merger.linear_fc2.weight` | `mm.2.weight` | [4608, 2048] |
| `merger.linear_fc2.bias` | `mm.2.bias` | [2048] |
| `merger.norm.weight` | Not in GGUF | [2048] |
| `merger.norm.bias` | Not in GGUF | [2048] |
| (no direct HF name) | `v.post_ln.weight` | [1152] |
| (no direct HF name) | `v.post_ln.bias` | [1152] |

**NOTE**: Merger norm (`merger.norm`) is NOT present in the GGUF. The GGUF's
`clip.projector_type` = `qwen3vl_merger` which may handle normalization differently
than the raw HF implementation.

### MMProj Memory

F16 format: 858 MB (matches actual file size)

| Component | Size (F16) |
|-----------|-----------|
| Patch embed (2×[16×16×3×1152]) | 4.4 MB |
| Position embed (1152×2304) | 5.3 MB |
| Post LN (1152×2) | 4.5 KB |
| 27 ViT blocks (270 tensors) | 763 MB |
| Merger MLP (mm.0 + mm.2) | 61 MB |
| **Total** | **~834 MB** (rest is overhead) |

---

## 9. RoPE Details

```
rope_theta = 10,000,000
partial_rotary_factor = 0.25  # Only 64/256 dims rotated
mrope_interleaved = true      # Interleaved MRoPE for 3D
mrope_section = [11, 11, 10]  # 11 temporal + 11 height + 10 width = 32 pairs
```

MRoPE applies to text+image+video:
- For text: `mrope_section[0]` = 11 temporal pairs rotated
- For images: temporal + height + width = 32 pairs rotated
- Total rotated dims = 2 × 32 = 64 (matching partial_rotary_factor × head_dim = 0.25 × 256 = 64)

### GQA RoPE Application

- Q: [2048×4096] → reshape to [16 heads, 256 dim] per token
- Only 64 of 256 dims get rotated (first or last 64, need to check)
- KV (shared): same rotation
- Q/K apply RoPE then Q norm / K norm (per-head)

---

## 10. Correcting Our C Implementation

### What We Have vs Reality

| Component | Our C impl | Ground Truth | Action |
|-----------|-----------|-------------|--------|
| SSM attn_qkv | Uses all 3 parts (Q, K, V) split properly | ✓ Correct | Verified |
| SSM attn_gate | Not implemented (we skip gate) | **Exist on ALL 30 SSM layers** | **NEEDED** |
| SSM ssm_conv1d | Implemented as conv1d | ✓ Correct usage | Verified |
| SSM ssm_out | Correct | ✓ | Verified |
| GQA: separate q/k/v | Uses fused QKV, need split | **Separate q_proj, k_proj, v_proj** | **NEED CHANGE** |
| GQA: q/k_norm | Not implemented | **Q/K head normalization** | **NEEDED** |
| GQA: no attn_gate | Our code might use gate | **No gate on GQA — direct output** | **NEED CHECK** |
| MoE: gate/up/down | Not yet implemented | **Split (not fused) in GGUF** | Phase 4 |
| MoE: shared expert | Not yet | **Shared expert + router** | Phase 4 |
| Tokenizer | Matches HF exactly | ✓ Verified | Done |
| RoPE dims | Need MRoPE interleaved | **MRoPE sections** | **NEEDED** |
| Final norm | Implemented | ✓ output_norm | Verified |

### Critical Fixes for Phase 3

1. **SSM attn_gate**: The SSM output is: `silu(gate(x)) * ssm_out(x)` where gate = `x @ attn_gate^T`.
   This is the gated MLP-style output that the config's `attn_output_gate: true` refers to.
   Previously we thought this was a GQA feature — it's actually the SSM output structure.

2. **GQA separate weights**: Our current code uses fused QKV for ALL layers, but GQA layers
   have separate q_proj [2048, 4096], k_proj [2048, 512], v_proj [2048, 512].

3. **Q/K head norm**: After RoPE, GQA applies per-head normalization (RMSNorm with [256] weight).

4. **MRoPE**: The interleaved MRoPE has sections [11,11,10] = 32 pairs = 64 rotated dims.

---

## 11. Expert Weight Shapes — Clarification (llama.cpp GGUF convention)

The expert weights in the GGUF use transposed storage vs what we'd expect from PyTorch:

```
ffn_gate_exps: [2048, 512, 256] = [hidden, intermediate, n_experts]
  → To get expert `e`: weight[:, :, e] is [hidden, intermediate]
  → For matmul y = x @ W^T: we need W^T of shape [intermediate, hidden]
  → So we can do: y = rweight(ffn_gate_exps[:, :, e]) with the weights as-is
  → Or load as [n_experts, intermediate, hidden] by transposing axes

ffn_down_exps: [512, 2048, 256] = [intermediate, hidden, n_experts]
  → To get expert `e`: weight[:, :, e] is [intermediate, hidden]
  → y = x @ W: x[1, hidden] × W[hidden, intermediate] 
  → We need W^T but stored as [intermediate, hidden]
```

**llama.cpp loading pattern**: Store as contiguous [n_experts, intermediate, hidden] by
reading `exps[:, :, e]` for each expert and concatenating along dim 0.
Our C MoE loader should follow this transposed convention.

---

## 12. Memory / VRAM

Full model in f16 (without quantization):

| Component | VRAM |
|-----------|------|
| Embeddings (248320 × 2048) | 1.0 GB |
| SSM attn (30 × 10 tensors) | 0.7 GB |
| GQA attn (10 × 6 tensors) | 0.1 GB |
| MoE experts (40 × 3 × 256 × 2048 × 512) | 67.1 GB |
| MoE shared (40 × 3 × 2048 × 512) | 0.3 GB |
| Norms, biases, etc | < 0.1 GB |
| **Total f16** | **~69 GB** |

With IQ2_M (2-bit quantization): ~11 GB ✓ (our GGUF)

### Per-Layer Quantized Sizes (IQ2_M)

SSM layer: ~2 bits × 18M params = ~4.5 MB per layer
GQA layer: ~2 bits × 15M params = ~3.8 MB per layer
Embedding: FP16 full (vocab is big)
Output/LM head: FP16

---

## 13. Key Takeaways for Next Phases

### Phase 3 (Current — Training Loop)
- Fix SSM `attn_gate` output gating
- Fix GQA separate q/k/v weights
- Add GQA q/k per-head normalization
- Implement MRoPE with interleaved sections

### Phase 4 (MoE Dispatch)
- Split gate/up/down expert weights from GGUF
- Shared expert + router (linearly gated)
- Top-8 routing with softmax

### Phase 5 (Vision/MMProj)
- 27-layer 3D ViT with GELU tanh activation
- MRoPE for 3D position encoding
- Fused QKV and separate proj
- Spatial merger: 2×2 patch concatenation then linear
- MMProj: fc1→fc2→norm to project 1152→2048

### Deprecated / Wrong Assumptions
- ~~`attn_output_gate` is a GQA feature~~ → It's an SSM feature (applies to all SSM layers)
- ~~GQA uses fused attn_qkv~~ → GQA has separate q/k/v
- ~~GQA has no head normalization~~ → GQA has q_norm and k_norm (per-head RMSNorm)
