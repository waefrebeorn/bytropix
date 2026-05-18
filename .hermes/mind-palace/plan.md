# Plan — May 18, 2026 — HONEST PATH TO 1:1 PARITY

## Phase 0: Verify SSM/GQA Forward Architecture
**Root cause is in the forward math, NOT quantization.**

### Task 0.1: Generate per-layer reference from llama.cpp
- Modify qwen35moe.cpp to dump hidden state after each layer's residual add
- OR use existing dump_ref_layers tool
- Compare: our_layer_N.bin vs ref_layer_N.bin for N=0..39
- Find the first layer where cos-sim < 0.99

### Task 0.2: Audit SSM Layer 0 forward against qwen3next.cpp
Files to compare:
- ~/llama.cpp/src/models/qwen3next.cpp (build_layer_attn_linear, build_qkvz)
- ~/llama.cpp/src/models/delta-net-base.cpp (build_delta_net_autoregressive)
- ~/llama.cpp/src/models/qwen35moe.cpp (our actual reference, same code)

Check:
- attn_qkv.weight split: Q[2048] + K[2048] + V[4096] = 8192 total
- attn_gate.weight: z_gate [2048, 4096] = silu(x @ W)
- Conv1d: kernel=4, depthwise on [d_inner + 2*n_group*d_state] channels
- Delta-net recurrence: state update formula
- ssm_out.weight: output projection after gated norm
- Residual: attn_out = ssm_out + input (before norm)

### Task 0.3: Audit GQA forward against qwen3next.cpp build_layer_attn
Check:
- attn_q.weight [2048,8192] = Q[4096] + gate[4096] split
- Q norm (RMSNorm per head, head_dim=256)
- K norm (RMSNorm per head)
- RoPE (64 dim, theta=10M, sections)
- Attention: Q @ K^T, softmax, weighted V sum
- Gate: sigmoid(gate) applied to attention output
- Output proj: attn_output.weight [4096,2048]

### Task 0.4: Verify Q6_K dequant against llama.cpp ggml-quants.c
- The block_q6_K struct may differ between quantized_dot_generic.c and reference
- Dump first block of ssm_out.weight from both, compare

## Phase 1: Fix Architecture Bugs
(Content depends on Phase 0 findings)

## Phase 2: Test With Quantized MoE
- The MoE quantized path I already wired should work once hidden states are correct
- Compare cos-sim before/after enabling quantized MoE

## Phase 3: Layer-by-layer verification
- Verify ALL 40 layers match within cos-sim 0.999
- Verify final logits match within cos-sim 0.999
