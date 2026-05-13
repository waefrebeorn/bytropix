# Phase 2: Attention Port — Gated DeltaNet → Hyperbolic Gyration

**Goal:** Port Qwen3.5+ Gated DeltaNet and GQA attention to C, with hyperbolic gyration
replacing the linear recurrence in DeltaNet layers.

**Depends on:** Phase 1 ✅ (GGUF reader + embeddings extracted)

## Target Architecture Per Layer

Two layer types, repeating every 4 layers:
```
Type A (3 of 4): [Gated DeltaNet] → [MoE]          ← 75% of compute
Type B (1 of 4): [GQA] → [MoE]                      ← 25% of compute
```

Key: MoE replaces SwiGLU FFN in Qwen3.5+ architecture for MoE layers.
Dense layers (Qwen3.5-9B) use SwiGLU FFN instead.

## GGUF Tensor Names (already parsed in Phase 1)

For each block N (0..39), the attention tensors are:
```
blk.N.attn_qkv.weight   → [2048, 8192]  Q8_K    — fused Q+K+V projection
blk.N.attn_gate.weight  → [2048, 4096]  Q8_K    — output gate projection
blk.N.attn_norm.weight  → [2048]        F32     — pre-attention RMSNorm
blk.N.ssm_conv1d.weight → [4, 8192]     F32     — local conv kernel (DeltaNet only)
blk.N.ssm_a             → [32]          F32     — DeltaNet recurrent coeff
blk.N.ssm_alpha.weight  → [2048, 32]    F32     — DeltaNet alpha projection
blk.N.ssm_beta.weight   → [2048, 32]    F32     — DeltaNet beta projection
blk.N.ssm_dt.bias       → [32]          F32     — DeltaNet delta t bias
blk.N.ssm_norm.weight   → [2048]        F32     — DeltaNet post-norm
blk.N.ssm_out.weight    → [4096, 2048]  F32     — DeltaNet output projection
```

After these, every block has MoE tensors:
```
blk.N.ffn_gate_inp.weight  → [2048, 256]  F32     — router
blk.N.ffn_gate_exps.weight → [2048, 512, 256] IQ2_XS — expert gate projections
blk.N.ffn_up_exps.weight   → [2048, 512, 256] IQ2_XS — expert up projections
blk.N.ffn_down_exps.weight  → [512, 2048, 256] IQ1_S  — expert down projections
blk.N.ffn_gate_shexp.weight → [2048, 512] Q8_K — shared expert gate
blk.N.ffn_up_shexp.weight   → [2048, 512] Q8_K — shared expert up
blk.N.ffn_down_shexp.weight → [512, 2048] Q8_K — shared expert down
blk.N.ffn_norm.weight       → [2048] F32 — pre-FFN norm
blk.N.post_attention_norm.weight → [2048] F32 — post-attention norm
```

**Critical insight:** The `attn_qkv.weight` is FUSED Q+K+V in one tensor, shape [2048, 8192].
8192 = 16 Q heads × 256 + 2 KV heads × 256 + 2 KV heads × 256 = 4096 + 512 + 512 = 5120?
No — 16 × 256 + 2 × 256 + 2 × 256 = 4096 + 512 + 512 = 5120. But the shape says 8192.
8192 = 16 × 256 + 2 × 256 + 2 × 256 + 16 × 128 + 32 × 128...
Actually: Q heads = 16 (full attention), KV heads = 2.
For LINEAR attention, we also have: 16 QK heads × 128 = 2048, 32 V heads × 128 = 4096.
2048 + 4096 = 6144 + 2048(Q_gate?) = 8192. So the QKV tensor contains BOTH the full
attention projections AND the linear attention projections fused together.

Breakdown of attn_qkv.weight [2048, 8192]:
  - 0..2047: Q_k (full attention)     — 16 heads × 128 dim (head_dim/2)
  - 2048..2559: K_k (full attention)   — 2 heads × 256 dim
  - 2560..3071: V_k (full attention)   — 2 heads × 256 dim
  - 3072..5119: Q_l (linear attention) — 16 heads × 128 dim (key)
  - 5120..8191: V_l (linear attention) — 32 heads × 128 dim (value)

## Step 2.1: Standard Gated DeltaNet (No Hyperbolic First)

**Reference:** Qwen3.6-35B-A3B config.json and GGUF tensor names:
```
ssm_conv_kernel: 4
ssm_state_size: 128 (linear_key_head_dim = key_head_dim = value_head_dim)
ssm_inner_size: 4096 (num_value_heads × value_head_dim = 32 × 128)
ssm_time_step_rank: 32
ssm_group_count: 16 (linear_num_key_heads)
attn_output_gate: true
head_dim: 256 (full attention head dim)
hidden_size: 2048
```

**Files:** `src/wubu_deltanet.c`, `include/wubu_deltanet.h`

```c
// Gated DeltaNet forward (standard, matching Qwen3.5 impl)
// Input: x[B, T, 2048]
// Output: y[B, T, 2048]

// Step 1: Fused QKV projection
float* qkv = x @ W_qkv;  // [B, T, 8192]

// Step 2: Split into linear attention (75% of layers) components
// The ssm_a, ssm_alpha, ssm_beta implement the DeltaNet recurrence:
//   h[t] = sigmoid(W_alpha @ x[t]) ⊙ h[t-1] + (1 - sigmoid) ⊙ (W_beta @ conv(x)[t])
// Simplified view:
//   z_t = sigmoid(alpha(x[t]))       // forget gate
//   h[t] = z_t ⊙ h[t-1] + (1-z_t) ⊙ conv1d(v)[t]
// But the actual implementation uses a structured SSM (Mamba-style):
//   A = -exp(ssm_a)  // negative exponential for stability
//   dt = softplus(W_dt @ x + bias_dt)
//   h[t] = exp(A*dt) ⊙ h[t-1] + (1 - exp(A*dt)) ⊙ conv(v)[t]
```

**Algorithm (SSM-style DeltaNet):**
```
// Qwen3.5 uses Mamba2-style SSM for the linear attention
v = x @ W_v          // [B, T, 4096] — from split of attn_qkv
v_conv = silu(conv1d(v, W_conv))  // [B, T, 4096] — depthwise conv, kernel=4
z = sigmoid(alpha)   // gate from ssm_alpha projection

// Recurrence (simplified — actually SSM with dt):
// h[t] = A_bar ⊙ h[t-1] + B_bar ⊙ v_conv[t]
// where A_bar = exp(-exp(ssm_a) * softplus(W_dt @ x + dt_bias))
//       B_bar = 1 - A_bar (for "supplement" gating; or separate z from alpha)

h[0] = 0
for t in 1..T:
    dt = softplus(W_dt @ x[t] + dt_bias)           // [32] time step
    A_bar = exp(-exp(ssm_a) * dt)                   // [32] continuous-time decay
    B_bar = z[t]                                     // [32] or derived from alpha
    h[t] = A_bar ⊙ h[t-1] + B_bar ⊙ v_conv[t]      // [4096]

// Output projection
o = h @ W_out       // [B, T, 4096] → [B, T, 2048] via W_out[4096, 2048]
g = sigmoid(x @ W_gate)  // output gate [B, T, 2048]
y = g ⊙ norm(o)     // post-norm + gate (ssm_norm.weight)
```

**Important:** The exact Qwen3.5 Gated DeltaNet implementation uses a structured SSM
(selective scan, Mamba-like), NOT a simple DeltaNet. The `ssm_a`, `ssm_dt`, `ssm_alpha`,
`ssm_beta` tensors match the Mamba2 architecture template in llama.cpp's `ggml-ssm.h`.

## Step 2.2: Hyperbolic Gated DeltaNet

Replace the linear SSM recurrence with Möbius gyration in Poincaré ball:

```c
// Hyperbolic Gated DeltaNet
// h[0] = 0 (origin in Poincaré ball)
for t in 1..T:
    dt = softplus(W_dt @ x[t] + dt_bias)
    A_bar = exp(-exp(ssm_a) * dt)
    
    // Map v_conv to Poincaré ball
    v_ball = exp_map(v_conv[t], R)
    
    // Linear part stays Euclidean (decay) — scale in tangent space
    // h_prev in ball → log_map → scale → exp_map → Möbius add
    h_prev_tangent = log_map(h[t-1], R)
    h_prev_tangent = A_bar ⊙ h_prev_tangent  // Euclidean decay in tangent space
    h_decayed = exp_map(h_prev_tangent, R)
    
    // Gyration addition: h[t] = h_decayed ⊕ v_ball
    // Möbius addition: x ⊕ y = ((1+2⟨x,y⟩+||y||²)x + (1-||x||²)y) / (1+2⟨x,y⟩+||x||²||y||²)
    h[t] = mobius_add(h_decayed, v_ball)

o = log_map(h, R) @ W_out    // Project back to tangent → output
g = sigmoid(x @ W_gate)
y = g ⊙ norm(o)
```

**Why this works:** The SSM recurrence is a linear combination in Euclidean space:
`h[t] = A ⊙ h[t-1] + B ⊙ v[t]`. In hyperbolic space via gyration, this becomes
a Möbius combination that preserves the metric. The `A_bar` decay happens in tangent
space (where Euclidean ops are valid) before mapping back.

## Step 2.3: GQA Full Attention (Full attention layers — every 4th)

Keep as standard softmax attention. The attn_qkv split for full layers:
```
Q = split(qkv, 0, 2048)   // [B, T, 16*128] — but wait, Q head dim is 256
```

Actually: for full attention layers (every 4th), the `attn_qkv.weight` shape [2048, 8192]
carries the SAME weight tensor, but the attention IS different. The `layer_types` array
determines which attention type runs, NOT the weights themselves. Qwen3.5 uses the SAME
QKV weights but applies full softmax attention in some layers and linear scan attention
in others.

This means: `attn_qkv` contains projections for BOTH attention types. The split is:
- Q_full: [2048, 16*256] = [2048, 4096] — first 4096 dims of qkv output
- K_full: [2048, 2*256] = [2048, 512] — next 512 dims  
- V_full: [2048, 2*256] = [2048, 512] — next 512 dims
- Q_linear: [2048, 16*128] = [2048, 2048] — next 2048 dims
- V_linear: [2048, 32*128] = [2048, 4096] — last 4096 dims

Total: 4096 + 512 + 512 + 2048 + 4096 = 11264 ≠ 8192. This doesn't add up.
Let me reconsider: the head_dim for full attention is 256 per head. 16 Q heads = 4096.
2 KV heads × 256 = 512 each. So Q+K+V = 4096+512+512 = 5120.
For linear: 16 QK heads × 128 = 2048, 32 V heads × 128 = 4096.
Total: 5120 + 2048 + 4096 = 11264. But shape is [2048, 8192].

This means the `attn_qkv` weight only covers SOME of the projections. The remaining
projections (like `ssm_out.weight` [4096, 2048] for DeltaNet output) are separate tensors.

**Resolution:** Need to trace through llama.cpp's `qwen35` model implementation to understand
the exact tensor split. The `attn_qkv.weight` likely only carries:
- Full attention Q (16×256=4096) 
- Full attention K (2×256=512)
- Full attention V (2×256=512)
Total: 5120, but shape says 8192. The extra 3072 dims must be for the linear attention QK projection
(16×128=2048 + something else = 3072? Let me check: 2048 + 1024? Or maybe V is 32×128=4096?)

Actually, 4096+512+512+2048+1024 = 8192. But 1024 = 8×128 (half of 16). 
Or: 4096(Q_full) + 512(K_full) + 512(V_full) + 2048(Q_linear_QK) = 7168. Still 1024 short.
Or: Q_linear = 2048 (for 16×128), V_linear first half = 1024 (for 8×128)... no.

**Must verify by reading llama.cpp source for qwen35 model.** The exact split affects
implementation correctness. This is a critical dependency for Phase 2.

## Step 2.4: Test on Baseline

1. Read attention weights from GGUF using Phase 1's gguf_reader
2. Implement forward pass matching Python reference (use tiny test)  
3. Compare output at each step against llama.cpp's output (one forward pass)
4. Then add hyperbolic gyration
5. Compare loss curves

## Success Criteria
- [ ] Gated DeltaNet forward pass matches llama.cpp output (within Q5_K quantization error)
- [ ] Hyperbolic version produces finite values (no NaN)
- [ ] Training speed >100 tok/s with CUDA kernels (or >10 tok/s CPU for prototype)
- [ ] Möbius exp/log maps are numerically stable (no norms > 0.99)

## Files to Create
```
src/wubu_deltanet.c            — SSM-style Gated DeltaNet + hyperbolic variant
include/wubu_deltanet.h        — Header
src/wubu_mobius.c              — Möbius addition + gyration ops
include/wubu_mobius.h          — Header
src/wubu_gqa.c                 — Full GQA attention (standard)
include/wubu_gqa.h             — Header
test_deltanet_forward.c        — Verify against Python/llama.cpp reference
```

## Pitfalls
1. **Linear recurrence is sequential** — the SSM recurrence `h[t] = f(h[t-1], x[t])` is
   O(T) sequential, not parallelizable across time. For short sequence (<4096), standard
   attention is actually faster. Need CUDA parallel scan for long sequences.
   - Fix: Use associative scan (Blelloch prefix sum) for CUDA kernel.
   - For CPU prototype: sequential loop is fine (T=512 max).

2. **Poincaré boundary instability** — norms near 1.0 cause tanh/artanh blowup.
   - Fix: Clamp norms to 0.99 in exp_map. Our embedding norms are 0.30-0.34 after
     mapping, so this won't trigger during normal forward passes. Only during training
     if gradients push embeddings outside.

3. **attn_qkv split unknown** — The exact split of the fused QKV tensor must be
   determined from llama.cpp source. Without this, we can't load pretrained weights.
   - Fix: Read `llama.cpp/src/models/qwen35.cpp` or `ggml-ssm.h` for the split.

4. **SSM vs DeltaNet discrepancy** — The Qwen3.5 implementation uses a structured SSM
   (Mamba2), not the simple DeltaNet described in the DeltaNet paper. The SSM has
   additional parameters (dt, A) that the simple DeltaNet doesn't.
   - Fix: Implement the SSM version first. The hyperbolic extension is the same
     regardless of whether it's DeltaNet or SSM — replace linear recurrence with gyration.
