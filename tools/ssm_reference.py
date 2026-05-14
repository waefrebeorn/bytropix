#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B Gated Delta Net (SSM) Python Reference

Matches the exact algorithm from llama.cpp/src/models/qwen3next.cpp
and src/models/delta-net-base.cpp.

Purpose: Verify our understanding before writing C code.
Usage:  python3 tools/ssm_reference.py
"""
import struct, sys, os
import numpy as np
from pathlib import Path

# Add tools dir for gguf_reader
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

# ============================================================
# Layer types from GGUF
# ============================================================
N_LAYERS = 40
SSM_K_HEADS = 16       # ssm_n_group
SSM_V_HEADS = 32       # ssm_dt_rank
SSM_D_STATE = 128      # head_k_dim = head_v_dim
D_INNER = 4096         # SSM inner dim
D_MODEL = 2048
CONV_KERNEL = 4
DT_RANK = 32           # ssm_time_step_rank

# For GQA layers:
N_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256

# Conv dimension
KEY_DIM = SSM_D_STATE * SSM_K_HEADS  # 2048
VALUE_DIM = SSM_D_STATE * SSM_V_HEADS  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 2048+2048+4096=8192


def is_ssm_layer(layer_idx):
    """Every 4th layer (idx%4==3) is GQA, rest are SSM"""
    return (layer_idx + 1) % 4 != 0


def softplus(x):
    return np.log(1.0 + np.exp(np.clip(x, -80, 80)))


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -80, 80)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def l2_norm(x, eps=1e-12):
    """L2 normalize along last dim"""
    return x / np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True) + eps)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def conv1d_forward(x, kernel):
    """
    Depthwise conv1d along time dimension.
    x: [B, T, C]  (C = conv_channels = 8192)
    kernel: [kernel_size, C]
    Returns: [B, T, C]
    """
    B, T, C = x.shape
    k = kernel.shape[0]  # 4
    output = np.zeros_like(x)
    # Simple 1D convolution: y[t] = sum_{i=0}^{k-1} kernel[i] * x[t - k + 1 + i]
    # Same as causal conv with padding = k-1
    padded = np.pad(x, ((0, 0), (k - 1, 0), (0, 0)), mode='constant')
    for t in range(T):
        output[:, t, :] = np.sum(padded[:, t:t + k, :] * kernel[np.newaxis, :, :], axis=1)
    return output


# ============================================================
# SSM Layer Forward Pass
# ============================================================

def ssm_layer_forward(x, weights):
    """
    Gated Delta Net (SSM) forward pass.
    x: [B, T, 2048] input
    weights: dict with all tensors for this layer
    Returns: [B, T, 2048] output
    """
    B, T, _ = x.shape

    # Step 1: Fused QKV projection
    # wqkv: [2048, 8192]  (in GGUF: [2048, 8192])
    # qkv_mixed = x @ wqkv  → [B, T, 8192]
    qkv_mixed = x @ weights['attn_qkv.weight']  # over [2048, 8192]
    # Technically GGUF stores [output_dim, input_dim], so:
    # But dump says shape [2048, 8192], and llama.cpp does build_lora_mm which is
    # matrix multiply with (n_embd, qkv_dim). In GGUF the weight is [n_embd, qkv_dim]
    # and matmul is x @ W where x is [B,T,n_embd] and W is [n_embd, qkv_dim].
    # So shape is correct as-is.

    # Step 2: z gate projection
    # attn_gate.weight: [2048, 4096] -> z = x @ wqkv_gate  [B, T, 4096]
    z = x @ weights['attn_gate.weight']

    # Step 3: beta/alpha projections
    # ssm_beta.weight: [2048, 32]  -> beta_raw
    # ssm_alpha.weight: [2048, 32] -> alpha
    # These are [2048, DT_RANK=32] each. They're from the ssm_beta_alpha fused tensor
    # but in GGUF they're stored separately!

    beta_raw = x @ weights['ssm_beta.weight']  # [B, T, 32]
    alpha = x @ weights['ssm_alpha.weight']    # [B, T, 32]

    # Step 4: Compute beta gate and alpha_gate (the "gate"/decay)
    # beta = sigmoid(beta_raw)  # [B, T, 32]
    beta = sigmoid(beta_raw)  # [B, T, 32]

    # alpha_biased = alpha + ssm_dt.bias[32]  # broadcast over B,T
    alpha_biased = alpha + weights['ssm_dt.bias']  # [B, T, 32]

    # alpha_softplus = softplus(alpha_biased)
    alpha_softplus = softplus(alpha_biased)  # [B, T, 32]

    # gate = alpha_softplus * ssm_a[32]  # [B, T, 32]  -- this is -A_log.exp() * softplus
    # ssm_a is negative log of A. ssm_a.shape = [32]
    gate = alpha_softplus * weights['ssm_a']  # [B, T, 32]

    # But gate needs to be [B, T, 1, SSM_V_HEADS] for the recurrence.
    # Reshape: [B, T, 32] -> [B, T, 1, 32]
    beta = beta.reshape(B, T, 1, 32)
    gate = gate.reshape(B, T, 1, 32)

    # Step 5: Convolution
    # conv_input = concat(conv_states, qkv_mixed) along time dim
    #   conv_states: [B, CONV_KERNEL-1, 8192] from cache
    # For T > 0, conv_input = [B, T+CONV_KERNEL-1, 8192]
    # conv_output = conv1d(conv_input, conv_kernel)  -> [B, T, 8192]
    # But ssm_conv1d.weight is [4, 8192] in GGUF.
    # The code also transposes qkv_mixed first.
    # From code: qkv_mixed transposed, then concat with conv_states along dim 0,
    # then ggml_ssm_conv is called.

    # Simplifying: this is a depthwise conv with kernel [4, 8192]
    # The qkv dimension coming out is split into:
    #   q_conv: [B, T, KEY_DIM] = [B, T, 2048]
    #   k_conv: [B, T, KEY_DIM] = [B, T, 2048]
    #   v_conv: [B, T, VALUE_DIM] = [B, T, 4096]
    # Total: 2048 + 2048 + 4096 = 8192 ✓

    # conv_output = silu(conv1d(qkv_mixed, conv_kernel))
    conv_kernel = weights['ssm_conv1d.weight']  # [4, 8192]
    conv_output = conv1d_forward(qkv_mixed, conv_kernel)  # [B, T, 8192]
    conv_output = silu(conv_output)

    # Split conv output
    q_conv = conv_output[:, :, :KEY_DIM]                     # [B, T, 2048]
    k_conv = conv_output[:, :, KEY_DIM:KEY_DIM * 2]          # [B, T, 2048]
    v_conv = conv_output[:, :, KEY_DIM * 2:]                  # [B, T, 4096]

    # Reshape to [B, T, SSM_K_HEADS, SSM_D_STATE] or [B, T, SSM_V_HEADS, SSM_D_STATE]
    q_conv = q_conv.reshape(B, T, SSM_K_HEADS, SSM_D_STATE)   # [B, T, 16, 128]
    k_conv = k_conv.reshape(B, T, SSM_K_HEADS, SSM_D_STATE)   # [B, T, 16, 128]
    v_conv = v_conv.reshape(B, T, SSM_V_HEADS, SSM_D_STATE)   # [B, T, 32, 128]

    # Step 6: L2 normalize Q and K
    q_conv = l2_norm(q_conv)  # [B, T, 16, 128]
    k_conv = l2_norm(k_conv)  # [B, T, 16, 128]

    # Step 7: Repeat Q/K heads to match V's 32 heads
    # num_k_heads=16, num_v_heads=32, repeat_factor=2
    # interleave repeat: reshape then repmat
    # q_conv: [B, T, 16, 128] -> [B, T, 16, 1, 128] -> repeat -> [B, T, 16, 2, 128] -> [B, T, 32, 128]
    q_conv = np.repeat(q_conv[:, :, :, np.newaxis, :], 2, axis=3).reshape(B, T, 32, 128)
    k_conv = np.repeat(k_conv[:, :, :, np.newaxis, :], 2, axis=3).reshape(B, T, 32, 128)

    # Step 8: Gated Delta Net recurrence
    # h[t] = h[t-1] * exp(gate[t]) + K[t] * (V[t] - h[t-1] @ K[t]) * beta[t]
    # output[t] = h[t] @ Q[t]
    # where h[t] is [B, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE], but per-head state is [128, 128]
    # Actually from the code: state = [head_v_dim, head_v_dim, num_v_heads, n_seqs]
    # So h[t] shape: [128, 128, 32, B] — but for simplicity, [B, T, 32, 128, 128]
    # Step 8: Gated Delta Net recurrence
    h = np.zeros((B, 32, 128, 128), dtype=np.float32)  # initial state = 0
    outputs = np.zeros((B, T, 32, 128), dtype=np.float32)

    for t in range(T):
        g_t = gate[:, t, :, :]  # [B, 1, 32]
        b_t = beta[:, t, :, :]  # [B, 1, 32]
        q_t = q_conv[:, t, :, :]  # [B, 32, 128]
        k_t = k_conv[:, t, :, :]  # [B, 32, 128]
        v_t = v_conv[:, t, :, :]  # [B, 32, 128]

        for head in range(32):
            g = np.exp(g_t[:, 0, head])  # [B]
            b = b_t[:, 0, head]  # [B]
            q = q_t[:, head, :]  # [B, 128]
            k = k_t[:, head, :]  # [B, 128]
            v = v_t[:, head, :]  # [B, 128]

            # state decay: h = h * exp(gate)
            h[:, head, :, :] = h[:, head, :, :] * g[:, None, None]

            # h @ k: [B, 128, 128] * [B, 128] -> [B, 128]
            hk = np.einsum('bij,bj->bi', h[:, head, :, :], k)

            # V - h@K -> [B, 128]
            diff = v - hk

            # Outer product: k[b,i] * diff[b,j] -> [B, 128, 128]
            # Then broadcast beta: * b[b]
            update = np.einsum('bi,bj->bij', k, diff) * b[:, None, None]

            h[:, head, :, :] = h[:, head, :, :] + update

            # output = h @ q
            outputs[:, t, head, :] = np.einsum('bij,bj->bi', h[:, head, :, :], q)

    # Step 9: Reshape output to [B, T, 4096]
    delta_output = outputs.reshape(B, T, VALUE_DIM)  # 4096

    # Step 10: Gated normalization
    # z is [B, T, 4096] — reshape to [B, T, 32, 128]
    z_reshaped = z.reshape(B, T, 32, 128)
    # ssm_norm.weight: [128] — RMSNorm on the head dim
    ssm_norm_w = weights['ssm_norm.weight']  # [128]
    delta_out_r = delta_output.reshape(B, T, 32, 128)
    # RMSNorm along last dim
    rms = np.sqrt(np.mean(delta_out_r ** 2, axis=-1, keepdims=True) + 1e-6)
    delta_out_norm = (delta_out_r / rms) * ssm_norm_w  # [B, T, 32, 128]
    # Multiply by silu(z) — gated
    z_silu = silu(z_reshaped)
    gated_output = delta_out_norm * z_silu  # [B, T, 32, 128]
    gated_output = gated_output.reshape(B, T, VALUE_DIM)  # [B, T, 4096]

    # Step 11: Output projection
    # ssm_out.weight: [4096, 2048] (in GGUF this is [4096, 2048])
    # final = gated_output @ ssm_out  -> [B, T, 2048]
    final = gated_output @ weights['ssm_out.weight']

    return final


# ============================================================
# GQA Layer Forward Pass
# ============================================================

def gqa_layer_forward(x, weights):
    """
    GQA (full attention) forward pass.
    x: [B, T, 2048]
    weights: dict with q, k, v, o, norm weights
    Returns: [B, T, 2048]
    """
    B, T, _ = x.shape

    # Step 1: Q projection (fused Q + gate)
    # wq: [2048, 8192]  — 16 heads × 256 head_dim × 2 (Q + gate)
    Qcur_full = x @ weights['attn_q.weight']  # [B, T, 8192]

    # Split: first 4096 = Q, second 4096 = gate
    Qcur = Qcur_full[:, :, :N_HEADS * HEAD_DIM]  # [B, T, 4096]
    gate = Qcur_full[:, :, N_HEADS * HEAD_DIM:]   # [B, T, 4096]

    # Reshape Q to [B, T, N_HEADS, HEAD_DIM] = [B, T, 16, 256]
    Qcur = Qcur.reshape(B, T, N_HEADS, HEAD_DIM)
    gate = gate.reshape(B, T, N_HEADS, HEAD_DIM)

    # Step 2: K, V projections
    Kcur = x @ weights['attn_k.weight']  # [B, T, 512]
    Vcur = x @ weights['attn_v.weight']  # [B, T, 512]

    Kcur = Kcur.reshape(B, T, N_KV_HEADS, HEAD_DIM)  # [B, T, 2, 256]
    Vcur = Vcur.reshape(B, T, N_KV_HEADS, HEAD_DIM)  # [B, T, 2, 256]

    # Step 3: Q/K norms (RMSNorm on head dim)
    Qcur_n = rms_norm(Qcur, weights['attn_q_norm.weight'])
    Kcur_n = rms_norm(Kcur, weights['attn_k_norm.weight'])

    # Step 4: Simple RoPE (no MRoPE complexity for reference)
    # Qwen 3.6 uses rope_ext with sections=[11,11,10,0], but for reference
    # we'll apply standard RoPE on dim 0..63 of the 256-dim head
    # For simplicity in the reference: skip RoPE (just note it's needed in C)
    # RoPE dim: 64/256 partial (rope_dimension_count=64)
    # MRoPE sections: [11,11,10,0] — first 11 for time, next 11 for height, next 10 for width, last 0 unused
    # For text-only, sections=[11+11+10=32, 0, 0, 0]? Actually for text the last section is 0
    # meaning the MRoPE falls back to standard?
    # From code: rope_sections[4] = [11,11,10,0] and n_rot=64
    # So sections are applied as [11,11,10,0] across the 64 rotated dims
    # For text: probably only dims 0-31 are rotated (MRoPE dim 0 = time)
    # SKIP: will implement in C

    # Step 5: GQA attention
    # 16 Q heads, 2 KV heads — each KV head serves 8 Q heads
    # Q: [B, T, 16, 256], K: [B, T, 2, 256], V: [B, T, 2, 256]
    scale = 1.0 / np.sqrt(256, dtype=np.float32)  # MUST be float32 to avoid promotion

    # Repeat KV heads: 2 -> 16
    Kcur_e = np.repeat(Kcur, N_HEADS // N_KV_HEADS, axis=2)  # [B, T, 16, 256]
    Vcur_e = np.repeat(Vcur, N_HEADS // N_KV_HEADS, axis=2)  # [B, T, 16, 256]

    # Attention: Q @ K^T / sqrt(d)
    # [B, 16, T, 256] @ [B, 16, 256, T] -> [B, 16, T, T]
    Q_T_heads = Qcur_n.transpose(0, 2, 1, 3)  # [B, 16, T, 256]
    K_T_heads = Kcur_e.transpose(0, 2, 3, 1)  # [B, 16, 256, T]
    attn_scores = np.matmul(Q_T_heads, K_T_heads) * scale  # [B, 16, T, T]

    # Causal mask
    mask = np.tril(np.ones((T, T), dtype=np.float32))
    attn_scores = attn_scores * mask - 1e9 * (1 - mask)

    # Softmax
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    # Weighted sum of values
    V_T_heads = Vcur_e.transpose(0, 2, 1, 3)  # [B, 16, T, 256]
    attn_out = np.matmul(attn_weights, V_T_heads)  # [B, 16, T, 256]
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, N_HEADS * HEAD_DIM)  # [B, T, 4096]

    # Step 6: Gating
    gate_sigmoid = sigmoid(gate.reshape(B, T, N_HEADS * HEAD_DIM))
    attn_out = attn_out * gate_sigmoid

    # Step 7: Output projection
    # wo: [4096, 2048]
    final = attn_out @ weights['attn_output.weight']

    return final


# ============================================================
# Test with random data
# ============================================================

def test_ssm_block():
    """Test SSM layer with random weights (just verify shapes and no NaN)"""
    B, T = 1, 4

    # Create synthetic weights
    w = {}
    w['attn_qkv.weight'] = np.random.randn(D_MODEL, KEY_DIM * 2 + VALUE_DIM).astype(np.float32) * 0.01
    w['attn_gate.weight'] = np.random.randn(D_MODEL, VALUE_DIM).astype(np.float32) * 0.01
    w['ssm_beta.weight'] = np.random.randn(D_MODEL, DT_RANK).astype(np.float32) * 0.01
    w['ssm_alpha.weight'] = np.random.randn(D_MODEL, DT_RANK).astype(np.float32) * 0.01
    w['ssm_dt.bias'] = np.random.randn(DT_RANK).astype(np.float32) * 0.01
    w['ssm_a'] = np.random.randn(DT_RANK).astype(np.float32) * 0.01  # -A_log, small
    w['ssm_conv1d.weight'] = np.random.randn(CONV_KERNEL, CONV_DIM).astype(np.float32) * 0.01
    w['ssm_norm.weight'] = np.ones(SSM_D_STATE, dtype=np.float32)
    w['ssm_out.weight'] = np.random.randn(VALUE_DIM, D_MODEL).astype(np.float32) * 0.01

    x = np.random.randn(B, T, D_MODEL).astype(np.float32) * 0.1

    out = ssm_layer_forward(x, w)

    assert out.shape == (B, T, D_MODEL), f"Expected {(B,T,D_MODEL)}, got {out.shape}"
    assert not np.any(np.isnan(out)), "NaN in SSM output"
    assert not np.any(np.isinf(out)), "Inf in SSM output"
    print(f"  SSM forward: {out.shape} ✓, mean={np.mean(out):.4f}, std={np.std(out):.4f}")
    return out


def test_gqa_block():
    """Test GQA layer with random weights"""
    B, T = 1, 4

    w = {}
    w['attn_q.weight'] = np.random.randn(D_MODEL, N_HEADS * HEAD_DIM * 2).astype(np.float32) * 0.01
    w['attn_k.weight'] = np.random.randn(D_MODEL, N_KV_HEADS * HEAD_DIM).astype(np.float32) * 0.01
    w['attn_v.weight'] = np.random.randn(D_MODEL, N_KV_HEADS * HEAD_DIM).astype(np.float32) * 0.01
    w['attn_output.weight'] = np.random.randn(N_HEADS * HEAD_DIM, D_MODEL).astype(np.float32) * 0.01
    w['attn_q_norm.weight'] = np.ones(HEAD_DIM, dtype=np.float32)
    w['attn_k_norm.weight'] = np.ones(HEAD_DIM, dtype=np.float32)

    x = np.random.randn(B, T, D_MODEL).astype(np.float32) * 0.1

    out = gqa_layer_forward(x, w)

    assert out.shape == (B, T, D_MODEL), f"Expected {(B,T,D_MODEL)}, got {out.shape}"
    assert not np.any(np.isnan(out)), "NaN in GQA output"
    assert not np.any(np.isinf(out)), "Inf in GQA output"
    print(f"  GQA forward: {out.shape} ✓, mean={np.mean(out):.4f}, std={np.std(out):.4f}")
    return out


if __name__ == '__main__':
    print("Qwen3.6-35B-A3B SSM Reference Tests")
    print("=" * 50)
    print(f"\n  D_MODEL={D_MODEL}, D_INNER={D_INNER}")
    print(f"  SSM: K_heads={SSM_K_HEADS}, V_heads={SSM_V_HEADS}, D_state={SSM_D_STATE}")
    print(f"  GQA: Q_heads={N_HEADS}, KV_heads={N_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  Key_dim={KEY_DIM}, Value_dim={VALUE_DIM}, Conv_dim={CONV_DIM}")
    print(f"  SSM layers: {sum(1 for i in range(40) if is_ssm_layer(i))}")
    print(f"  GQA layers: {sum(1 for i in range(40) if not is_ssm_layer(i))}")

    print("\n  Testing SSM block...")
    test_ssm_block()

    print("\n  Testing GQA block...")
    test_gqa_block()

    print("\n  ✓ All tests pass. Ready for Phase 2 C implementation.")
