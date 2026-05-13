# Qwen3.6-35B-A3B (qwen3next) GGUF Tensor Layout — Complete Analysis

**Source:** `llama.cpp/src/models/qwen3next.cpp` + actual GGUF dump
**Model:** `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (11.5GB, 733 tensors)
**Architecture:** `qwen35moe` (tensor naming same as qwen35)

## Key Hyperparameters (from GGUF KV)

| Param | Value | Meaning |
|-------|-------|---------|
| `block_count` | 40 | 40 layers |
| `embedding_length` | 2048 | hidden dim |
| `attention.head_count` | 16 | full-attn Q heads |
| `attention.head_count_kv` | 2 | full-attn KV heads |
| `attention.key_length` | 256 | full-attn head dim |
| `attention.value_length` | 256 | full-attn head dim |
| `ssm.conv_kernel` | 4 | conv1d kernel size |
| `ssm.state_size` | 128 | SSM state dim |
| `ssm.group_count` | 16 | SSM num_k_heads |
| `ssm.time_step_rank` | 32 | dt bottleneck rank |
| `ssm.inner_size` | 4096 | SSM inner dim |
| `full_attention_interval` | 4 | every 4th layer is GQA |
| `rope.dimension_count` | 64 | partial RoPE dim |
| `rope.dimension_sections` | [11, 11, 10, 0] | MRoPE sections |
| `expert_count` | 256 | total experts |
| `expert_used_count` | 8 | top-k routed |

## Layer Type Pattern (40 layers)

```
index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16...
type:  SSM SSM SSM GQA SSM SSM SSM GQA SSM SSM SSM GQA SSM SSM SSM GQA SSM...
```

Every 4th layer (index 3, 7, 11, 15, 19, 23, 27, 31, 35, 39) is **GQA** (full attention).
All others (30 of 40) are **SSM** (Gated DeltaNet/Mamba2-style).

Confirmed: blocks divisible by 4 have `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`
All other blocks have `attn_qkv`, `attn_gate`, `ssm_*`, no `attn_q/k/v` separately.

## SSM Layer Tensors (blk.N where N % 4 != 0)

18 tensors per SSM layer:

| Tensor | Shape | Type | Purpose |
|--------|-------|------|---------|
| `attn_norm.weight` | [2048] | F32 | Pre-attention RMSNorm |
| `attn_qkv.weight` | [2048, 8192] | Q8_K | Fused Q+K+V projection |
| `attn_gate.weight` | [2048, 4096] | Q8_K | z gate projection (`wqkv_gate`) |
| `ssm_conv1d.weight` | [4, 8192] | F32 | Depthwise conv kernel |
| `ssm_alpha.weight` | [2048, 32] | F32 | α projection (dt compute) |
| `ssm_beta.weight` | [2048, 32] | F32 | β projection (gate) |
| `ssm_dt.bias` | [32] | F32 | dt bias (added to α) |
| `ssm_a` | [32] | F32 | -A_log (negative log decay) |
| `ssm_norm.weight` | [128] | F32 | Post-SSM RMSNorm head_dim |
| `ssm_out.weight` | [4096, 2048] | IQ1_S | SSM output projection |
| `post_attention_norm.weight` | [2048] | F32 | Post-attention norm |
| `ffn_gate_inp.weight` | [2048, 256] | F32 | Expert router |
| `ffn_gate_inp_shexp.weight` | [2048] | F32 | Shared expert gate |
| `ffn_gate_exps.weight` | [2048, 512, 256] | IQ2_XS | Expert gate proj |
| `ffn_up_exps.weight` | [2048, 512, 256] | IQ2_XS | Expert up proj |
| `ffn_down_exps.weight` | [512, 2048, 256] | IQ1_S | Expert down proj |
| `ffn_gate_shexp.weight` | [2048, 512] | Q8_K | Shared expert gate |
| `ffn_up_shexp.weight` | [2048, 512] | Q8_K | Shared expert up |
| `ffn_down_shexp.weight` | [512, 2048] | IQ1_S | Shared expert down |

NOTE: SSM layers lack `attn_q_norm`, `attn_k_norm` (those are for GQA only).
NOTE: ssm_out.type varies — most are IQ1_S, but later blocks may differ.

## GQA Layer Tensors (blk.N where N % 4 == 0)

18 tensors per GQA layer:

| Tensor | Shape | Type | Purpose |
|--------|-------|------|---------|
| `attn_norm.weight` | [2048] | F32 | Pre-attention RMSNorm |
| `attn_q.weight` | [2048, 8192] | Q8_K | Q + gate fused projection |
| `attn_q_norm.weight` | [256] | F32 | Q head norm |
| `attn_k.weight` | [2048, 512] | Q8_K | Key projection (2 heads × 256) |
| `attn_k_norm.weight` | [256] | F32 | K head norm |
| `attn_v.weight` | [2048, 512] | Q8_K | Value projection (2 heads × 256) |
| `attn_output.weight` | [4096, 2048] | Q8_K | Output projection |
| `post_attention_norm.weight` | [2048] | F32 | Post-attention norm |
| `ffn_gate_inp.weight` | [2048, 256] | F32 | Expert router |
| (same MoE tensors as SSM layers) | | | |

GQA layers do NOT have: `attn_gate`, `ssm_*`, `attn_qkv`.
EXCEPT: `blk.3`, `blk.7`, `blk.11`, etc. do NOT have `attn_qkv` — they have `attn_q`, `attn_k`, `attn_v` SEPARATELY.

## CRITICAL: attn_qkv.weight Split Resolution

**`attn_qkv.weight` shape = [2048, 8192] for SSM layers only.**

From `qwen3next.cpp` line 84:
```cpp
layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i),
    { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
```

Where:
- `head_k_dim = ssm_d_state = 128`
- `n_k_heads = ssm_n_group = 16`
- `n_v_heads = ssm_dt_rank = 32`
- `head_v_dim = d_inner / n_v_heads = 4096 / 32 = 128`
- `key_dim = head_k_dim * n_k_heads = 128 * 16 = 2048`
- `value_dim = head_v_dim * n_v_heads = 128 * 32 = 4096`
- `qkvz_dim = key_dim * 2 + value_dim * 2 = 4096 + 8192 = 12288` (for ssm_in, legacy)

But `wqkv` uses `key_dim * 2 + value_dim = 2048 * 2 + 4096 = 8192` ✓

### EXACT SPLIT of wqkv (attn_qkv) output:

```
output: [B, T, 8192]
├── Q (key_dim):           [B, T, 2048]   — linear attention query
├── K (key_dim):           [B, T, 2048]   — linear attention key
└── V (value_dim):         [B, T, 4096]   — linear attention value
```

Total: 2048 + 2048 + 4096 = 8192 ✓

BUT WAIT — the code at line 312 says:
```cpp
int64_t qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
= 2 * 128 + 2 * 128 * (32/16) = 256 + 2*128*2 = 256 + 512 = 768
```

This is for the LEGACY `ssm_in` tensor (NOT the current `wqkv`). The current path uses `wqkv` directly (line 297-306):
```cpp
if (model.layers[il].wqkv) {
    // optimized path
    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
    // qkv_mixed: [8192, n_seq_tokens, n_seqs]
```

Then in `build_layer_attn_linear`, qkv_mixed is transposed to `[n_seq_tokens, 8192, n_seqs]` and concatenated with conv_states before `ggml_ssm_conv` processes it.

### After Conv + SiLU Split (line 288-308):

```cpp
int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
= 128 * 16 * 2 + 128 * 32 = 4096 + 4096 = 8192
```

So the conv output dimension is also 8192, split as:
- `q_conv`: [head_k_dim=128, num_k_heads=16] = 2048 elements
- `k_conv`: [head_k_dim=128, num_k_heads=16] = 2048 elements
- `v_conv`: [head_v_dim=128, num_v_heads=32] = 4096 elements

## Convolution Details

```
conv_input = concat(conv_states, qkv_mixed) along dim 0
conv_kernel = [4, 8192]
conv_states: [conv_kernel_size-1=3, conv_channels=8192+2048=10240, n_seqs]
```

Wait — `conv_channels = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state`
= 4096 + 2 * 16 * 128 = 4096 + 4096 = 8192

So conv_input = [3+8192=8195, 8192, n_seqs], kernel=[4, 8192], output=[8192, 8192, n_seqs].

## alpha/beta Computation

From `ssm_beta_alpha.weight [2048, ba_dim=64]`:
```
ba_dim = n_v_heads * 2 = 32 * 2 = 64
```
Split: beta=[32], alpha=[32] (both divided by num_k_heads=16, re-expanded later)

```
beta = sigmoid(b)
alpha = reshape(a, [num_v_heads=32, n_seq_tokens, n_seqs])
alpha_biased = alpha + ssm_dt[32]
alpha_softplus = softplus(alpha_biased)
gate = alpha_softplus * ssm_a[32]  // -A_log.exp() * softplus ≈ exp(A*dt)
```

## z Gate

From `attn_gate.weight [2048, 4096]` where 4096 = value_dim:
```cpp
z = x @ wqkv_gate  // [B, T, 4096]
```
z is reshaped and used as the gate for `build_norm_gated(output, ssm_norm, z, il)`.

## Output Projection

```
final_output: [head_v_dim * num_v_heads=4096, n_seq_tokens, n_seqs]
cur = final_output @ ssm_out[4096, 2048]  →  [B, T, 2048]
```

## GQA Layer Attention (every 4th layer)

From `build_layer_attn` (lines 206-283):
```
Qcur_full = x @ wq           → [B, T, 8192]  // 16 heads × 256 × 2 = 8192
Split: Qcur[16×256=4096], gate[16×256=4096]

Kcur = x @ wk                 → [B, T, 512]   // 2 heads × 256
Vcur = x @ wv                 → [B, T, 512]   // 2 heads × 256

Qnorm = RMSNorm(Qcur) on head dim
Knorm = RMSNorm(Kcur) on head dim
Qrope = rope_ext(Qnorm, sections=[11,11,10,0])
Krope = rope_ext(Knorm, sections)

attn = softmax(Q @ K^T / sqrt(256))
gate_sigmoid = sigmoid(gate)
output = attn @ V * gate_sigmoid
output = output @ wo[4096, 2048]
```

## Key Differences from qwen35 (Phase 2 README)

1. **NO fused attn_qkv in GQA layers** — Q/K/V are separate `wq`, `wk`, `wv` tensors. The plan's analysis was WRONG about GQA using attn_qkv.
2. **SSM attn_qkv.weight output**: Q[2048], K[2048], V[4096] — NOT the messy split the plan described.
3. **Wqkv_gate IS attn_gate.weight**: [2048, 4096] — produces z gate for output gating.
4. **ssm_beta_alpha is fused**: [2048, 64] — split into beta[32] and alpha[32], NOT separate tensors. The GGUF has `ssm_beta.weight` and `ssm_alpha.weight` as separate tensors though!
5. **MRoPE for GQA only**: SSM layers have NO RoPE (use conv1d for positional info).
6. **L2 norm on Q/K (not RMSNorm)**: `ggml_l2_norm` used in SSM layers, not RMSNorm.
7. **No DeltaNet recurrence** — it's a Gated Delta Network with chunked parallel processing (not sequential).

## SSM Recurrence Formula (Exact)

The "recurrence" is NOT the Mamba2 scan. It's a **chunked linear attention** with exponential decay:

```
// For chunked (n_tokens > 1):
// Key computation with exponential decay mask:
decay_mask[i][j] = exp(alpha_softplus_ij * ssm_a - alpha_softplus_i * ssm_a)  for j ≤ i
attn = triu? Actually tril lower: ((K_beta) @ K) ⊙ decay_mask, then solve triangular system

// For autoregressive (n_tokens == 1):
state = state * exp(gate)       // decay
state += K * (V - state @ K) * beta   // update
output = state @ Q              // read
```

The actual math:
```
h[t] = h[t-1] * exp(gate[t]) + K[t] * (V[t] - h[t-1] @ K[t]) * beta[t]
output[t] = h[t] @ Q[t]
```

This is a **Gated Delta Network** — a type of linear attention with:
1. Gating via `beta = sigmoid(...)` — controls write-in
2. Exponential decay via `exp(-exp(ssm_a) * softplus(alpha + dt_bias))`
3. Recurrent state `h[t]` of shape [head_v_dim=128, head_v_dim=128] per head (32 heads)
4. Read via `h[t] @ Q[t]` — output is state times query

The chunked version batches this efficiently using triangular matrix operations.

## Impact on Phase 2 Implementation Plan

### SIMPLIFIED (much cleaner than expected):
1. **SSM Layer forward pass:**
   - `qkv = x @ wqkv` → Q[2048], K[2048], V[4096]
   - `conv1d(qkv)` → convolve, SiLU
   - Split conv output → `q_conv[128,16]`, `k_conv[128,16]`, `v_conv[128,32]`
   - L2 normalize Q_conv, K_conv
   - Repeat Q/K heads: 16→32 (to match V's 32 heads)
   - `alpha = x @ w_ssm_alpha` + `ssm_dt.bias` → `softplus` → `* ssm_a`
   - `beta = sigmoid(x @ w_ssm_beta)`
   - **Gated Delta Net**: `h[t] = h[t-1]*exp(gate) + K*(V - h[t-1]@K)*beta`
   - Output: `h[t] @ Q[t]`
   - Gated norm: `output * silu(z_gate)`, then `output @ ssm_out`

2. **GQA Layer forward pass:**
   - `Q = x @ wq` → split Q[16×256] and gate[16×256]
   - `K = x @ wk` → [2×256]
   - `V = x @ wv` → [2×256]
   - Q/K norm, RoPE (MRoPE sections=[11,11,10,0])
   - Standard GQA attention with sigmoid gate
   - Output proj

### WRONG ASSUMPTIONS CORRECTED:
- ❌ "attn_qkv contains both full and linear attention projections" — WRONG. attn_qkv is ONLY for SSM layers
- ❌ "Q heads = 16 for full attention, split complex" — Full attn uses SEPARATE wq[2048,8192], wk[2048,512], wv[2048,512] tensors
- ❌ "Mamba2-style SSM recurrence" — It's Gated Delta Net (chunked parallel linear attention), not sequential SSM scan
- ✅ "30 SSM, 10 GQA" — CORRECT
- ✅ "SSM inner_size = 4096" — CORRECT
- ✅ "Not DeltaNet [paper]" — Actually it IS a type of DeltaNet, just structured differently from the paper
