#!/usr/bin/env python3
"""Python reference: run BOS through our model architecture and dump layer 0.
Use the gguf library to read all weights (already dequantized), then implement
the same forward pass as our C code. Compare all intermediate values.
"""
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

# Load layer 0 SSM weights
D_MODEL = 2048
VOCAB  = 248320
SSM_K_HEADS = 16
SSM_V_HEADS = 32
SSM_D_STATE = 128
KEY_DIM = SSM_D_STATE * SSM_K_HEADS  # 2048
VALUE_DIM = SSM_D_STATE * SSM_V_HEADS  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 8192
CONV_KERNEL = 4
DT_RANK = 32

def load_tensor(name):
    t = [t for t in r.tensors if t.name == name][0]
    # GGUF tensor shapes from Python: [innermost, ..., outermost]
    # For the gguf library, the raw data is already dequantized
    # t.data gives a memory-mapped numpy array
    data = np.array(t.data)
    return data

# Get BOS embedding
tok_embd = load_tensor('token_embd.weight')
bos = 248044
emb = tok_embd[:, bos].copy()  # shape [D_MODEL]
print(f"Embedding: mean={emb.mean():.6f} std={emb.std():.6f}")

# Load SSM weights for layer 0
attn_qkv = load_tensor('blk.0.attn_qkv.weight')  # [D_MODEL=2048, CONV_DIM=8192]
attn_gate = load_tensor('blk.0.attn_gate.weight')  # [D_MODEL=2048, VALUE_DIM=4096]
ssm_beta = load_tensor('blk.0.ssm_beta.weight')  # [D_MODEL=2048, DT_RANK=32]
ssm_alpha = load_tensor('blk.0.ssm_alpha.weight')  # [D_MODEL=2048, DT_RANK=32]
ssm_dt_bias = load_tensor('blk.0.ssm_dt.bias')  # [32]
ssm_a = load_tensor('blk.0.ssm_a')  # [32]
ssm_conv1d = load_tensor('blk.0.ssm_conv1d.weight')  # [CONV_KERNEL=4, CONV_DIM=8192]
ssm_norm = load_tensor('blk.0.ssm_norm.weight')  # [128]
ssm_out = load_tensor('blk.0.ssm_out.weight')  # [VALUE_DIM=4096, D_MODEL=2048]
attn_norm_w = load_tensor('blk.0.attn_norm.weight')  # [2048]
post_attn_norm_w = load_tensor('blk.0.post_attention_norm.weight')  # [2048]

# Norms for all 40 layers would be needed for full pass
# For now, just do layer 0

def rms_norm(x, weight, eps=1e-6):
    """x: [d], weight: [d]"""
    rms = np.sqrt(np.mean(x**2) + eps)
    return x / rms * weight

# Step 0: Pre-attention RMSNorm
x = emb.copy()
normed = rms_norm(x, attn_norm_w)
print(f"Normed: mean={normed.mean():.6f} std={normed.std():.6f}")

# Step 1: QKV projection
# x: [D_MODEL], attn_qkv: [D_MODEL, CONV_DIM]
# qkv = x.T @ W where W is [D_MODEL, CONV_DIM]
# Our C code: qkv_s[j] = sum_i x_s[i] * w->attn_qkv_weight[i + j * D_MODEL]
# = x @ W where W[i, j] = weight[i + j*D_MODEL]
qkv_out = np.zeros(CONV_DIM)
for j in range(CONV_DIM):
    s = 0.0
    for i in range(D_MODEL):
        s += normed[i] * attn_qkv[i, j]  # attn_qkv[i, j] = element at inner=i, outer=j
    qkv_out[j] = s
print(f"QKV: mean={qkv_out.mean():.6f} std={qkv_out.std():.6f}")

# Step 2: Z gate projection
z_out = np.zeros(VALUE_DIM)
for j in range(VALUE_DIM):
    s = 0.0
    for i in range(D_MODEL):
        s += normed[i] * attn_gate[i, j]
    z_out[j] = s
print(f"Z: mean={z_out.mean():.6f} std={z_out.std():.6f}")

# Step 3: Beta/Alpha projections
beta_raw = np.zeros(DT_RANK)
alpha_raw = np.zeros(DT_RANK)
for j in range(DT_RANK):
    sb = 0.0
    sa = 0.0
    for i in range(D_MODEL):
        sb += normed[i] * ssm_beta[i, j]
        sa += normed[i] * ssm_alpha[i, j]
    beta_raw[j] = sb
    alpha_raw[j] = sa
print(f"Beta_raw: {beta_raw}")
print(f"Alpha_raw: {alpha_raw}")

# Step 4: Beta = sigmoid, Alpha = softplus(alpha_raw + dt_bias) * ssm_a
beta = 1.0 / (1.0 + np.exp(-beta_raw))
alpha_biased = alpha_raw + ssm_dt_bias
alpha_softplus = np.log(1.0 + np.exp(alpha_biased))
gate = alpha_softplus * ssm_a
print(f"Beta: {beta}")
print(f"Gate (exp part): {gate}")

# Step 5: Convolution
# Build conv_input: [T+CONV_KERNEL-1, C] with leading zeros as conv_state
T = 1
B = 1
conv_input = np.zeros((T + CONV_KERNEL - 1, CONV_DIM))
# conv_state (first CONV_KERNEL-1 elements) are zeros (initial state)
# rest is qkv_out
conv_input[CONV_KERNEL-1:, :] = qkv_out.reshape(1, CONV_DIM)

# Convolution: output[t, c] = sum_ki kernel[ki, c] * input[t+ki, c]
conv_out = np.zeros((T, CONV_DIM))
for ki in range(CONV_KERNEL):
    conv_out += ssm_conv1d[ki, :] * conv_input[ki:T+ki, :]
print(f"Conv: mean={conv_out.mean():.6f} std={conv_out.std():.6f}")

# SiLU
def silu(x):
    return x / (1.0 + np.exp(-x))
conv_silu = silu(conv_out[0])
print(f"Conv+SiLU: mean={conv_silu.mean():.6f} std={conv_silu.std():.6f}")

# Step 6: Split Q, K, V
q_conv = conv_silu[:KEY_DIM]
k_conv = conv_silu[KEY_DIM:2*KEY_DIM]
v_conv = conv_silu[2*KEY_DIM:]
print(f"Q: mean={q_conv.mean():.6f} std={q_conv.std():.6f}")
print(f"K: mean={k_conv.mean():.6f} std={k_conv.std():.6f}")
print(f"V: mean={v_conv.mean():.6f} std={v_conv.std():.6f}")

# Step 7: L2 Normalize Q and K
def l2_norm(x, n_groups, d_per_group, eps=1e-6):
    """x: [n_groups * d_per_group], L2 normalize per group"""
    out = np.zeros_like(x)
    for g in range(n_groups):
        gx = x[g*d_per_group:(g+1)*d_per_group]
        norm = np.sqrt(np.sum(gx**2) + eps)
        out[g*d_per_group:(g+1)*d_per_group] = gx / norm
    return out

q_norm = l2_norm(q_conv, SSM_K_HEADS, SSM_D_STATE)
k_norm = l2_norm(k_conv, SSM_K_HEADS, SSM_D_STATE)
print(f"Q norm: mean={q_norm.mean():.6f} std={q_norm.std():.6f}")

# Step 8: Repeat Q/K heads (cyclic repeat: vh % SSM_K_HEADS)
# For vh=0: kh=0, for vh=1: kh=1, ..., for vh=15: kh=15, for vh=16: kh=0, ...

# Step 9: Delta Net Recurrence (autoregressive, 1 token)
# State initialized to all zeros
state = np.zeros((SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE))

delta_out_all = np.zeros(VALUE_DIM)
s = 0  # single token index

for vh in range(SSM_V_HEADS):
    kh = vh % SSM_K_HEADS
    
    bg = beta[vh]  # beta per V-head
    gg = np.exp(gate[vh])  # exp(gate) per V-head
    
    # Get Q, K, V for this head
    q_vh = q_norm[kh * SSM_D_STATE:(kh+1) * SSM_D_STATE]
    k_vh = k_norm[kh * SSM_D_STATE:(kh+1) * SSM_D_STATE]
    v_vh = v_conv[vh * SSM_D_STATE:(vh+1) * SSM_D_STATE]
    
    # Scale Q by 1/sqrt(d)
    q_scale = 1.0 / np.sqrt(SSM_D_STATE)
    q_scaled = q_vh * q_scale
    
    # Get state for this head
    h = state[vh]
    
    # Step 8a: State decay
    h *= gg
    
    # Step 8b: h @ k
    hk = h @ k_vh  # [SSM_D_STATE]
    
    # Step 8c: diff = v - hk
    diff = v_vh - hk
    
    # Step 8d: State update
    h += np.outer(diff * bg, k_vh)  # H += k ⊗ diff * beta
    
    # Step 8e: Output = H @ q
    out_vh = h @ q_scaled
    
    delta_out_all[vh * SSM_D_STATE:(vh+1) * SSM_D_STATE] = out_vh

print(f"Delta out: mean={delta_out_all.mean():.6f} std={delta_out_all.std():.6f}")

# Step 10: Gated normalization
z_silu = silu(z_out)
delta_normed = np.zeros(VALUE_DIM)
for vh in range(SSM_V_HEADS):
    out_vh = delta_out_all[vh * SSM_D_STATE:(vh+1) * SSM_D_STATE]
    z_vh = z_silu[vh * SSM_D_STATE:(vh+1) * SSM_D_STATE]
    
    # RMSNorm per head
    rms = np.sqrt(np.mean(out_vh**2) + 1e-6)
    scale = 1.0 / rms
    delta_normed[vh * SSM_D_STATE:(vh+1) * SSM_D_STATE] = out_vh * scale * ssm_norm * z_vh

print(f"Delta normed: mean={delta_normed.mean():.6f} std={delta_normed.std():.6f}")

# Step 11: Output projection
# ssm_out: [VALUE_DIM=4096, D_MODEL=2048]
# result[j] = sum_i delta_normed[i] * ssm_out[i, j]
ssm_output = np.zeros(D_MODEL)
for j in range(D_MODEL):
    s = 0.0
    for i in range(VALUE_DIM):
        s += delta_normed[i] * ssm_out[i, j]  # ssm_out[i, j] = element at inner=i, outer=j
    ssm_output[j] = s
print(f"SSM output: mean={ssm_output.mean():.6f} std={ssm_output.std():.6f}")

# Save for comparison with C code
ssm_output.tofile('/tmp/py_ssm_output.bin')
delta_out_all.tofile('/tmp/py_delta_out.bin')
print("\nSaved Python outputs to /tmp/py_*.bin")
