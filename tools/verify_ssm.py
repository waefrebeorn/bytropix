#!/usr/bin/env python3
"""Verify SSM layer 0 step by step."""
import struct, math, sys, os
import numpy as np

def load(fn):
    with open(fn,'rb') as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.float32)

# Load normed input (the debug dump from our code)
normed = load('/tmp/dbg_qkv_input.bin').reshape(2, 2048)
print(f"normed shape: {normed.shape}")

# Load weights
qkv_w = load('/tmp/w0/attn_qkv.bin').reshape(2048, 8192, order='F')  # dims [2048,8192], col-major in numpy
gate_w = load('/tmp/w0/attn_gate.bin').reshape(2048, 4096, order='F')
beta_w = load('/tmp/w0/ssm_beta.bin').reshape(2048, 32, order='F')
alpha_w = load('/tmp/w0/ssm_alpha.bin').reshape(2048, 32, order='F')
dt_bias = load('/tmp/w0/ssm_dt.bin')
ssm_a = load('/tmp/w0/ssm_a.bin')
conv_w = load('/tmp/w0/ssm_conv1d.bin').reshape(4, 8192, order='F')
ssm_norm_w = load('/tmp/w0/ssm_norm.bin')
ssm_out_w = load('/tmp/w0/ssm_out.bin').reshape(4096, 2048, order='F')

print(f"qkv_w shape: {qkv_w.shape}")
print(f"ssm_a: {ssm_a}")

D_MODEL = 2048
C = 8192  # conv dim
K = 4     # conv kernel
SSM_D_STATE = 128
SSM_K_HEADS = 16
SSM_V_HEADS = 32
KEY_DIM = SSM_K_HEADS * SSM_D_STATE  # 2048
VALUE_DIM = SSM_V_HEADS * SSM_D_STATE  # 4096
DT_RANK = 32

N = 2  # tokens
B = 1
T = 2

def check(name, our_val, ref_val, tol=1e-4):
    """Compare our value with reference."""
    if our_val.shape != ref_val.shape:
        print(f"{name}: SHAPE MISMATCH our={our_val.shape} ref={ref_val.shape}")
        return False
    max_diff = np.max(np.abs(our_val - ref_val))
    cos_sim = np.dot(our_val.flatten(), ref_val.flatten()) / (
        np.linalg.norm(our_val) * np.linalg.norm(ref_val) + 1e-30)
    ratio = np.linalg.norm(our_val) / (np.linalg.norm(ref_val) + 1e-30)
    ok = max_diff < tol and cos_sim > 0.999
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: max_diff={max_diff:.6f} cos_sim={cos_sim:.6f} ratio={ratio:.4f}")
    if not ok and name != "output_proj":
        print(f"    our[0:5]: {our_val.flatten()[:5]}")
        print(f"    ref[0:5]: {ref_val.flatten()[:5]}")
    return ok

# ========== STEP 1: QKV Projection ==========
# x @ W_qkv where x shape is [T, D_MODEL], W_qkv shape is [D_MODEL, C]
# In row-major: output[t][j] = sum_i x[t][i] * W_qkv[i][j]
# W_qkv has dims [D_MODEL, C] = [2048, 8192]
# Element W[i][j] at offset i + j * 2048
# In numpy with order='F': qkv_w[i, j] = data[i + j*2048]
qkv_out = normed @ qkv_w  # [T, 8192]
print(f"\nStep 1: QKV projection")
print(f"  qkv_out[0,0:5]: {qkv_out[0,:5]}")

# ========== STEP 2: Conv1d ==========
# conv_input: [B, T+K-1, C]
# For B=1, T=2, K=4: conv_input has 2+4-1=5 timesteps
# First 3 = conv_state (zeros), last 2 = qkv_out
conv_input = np.zeros((5, C))
conv_input[3:5, :] = qkv_out  # first 3 are zeros (conv_state)

conv_out = np.zeros((2, C))
for t in range(2):
    for c in range(C):
        s = 0.0
        for ki in range(4):
            t_in = t + ki
            s += conv_input[t_in, c] * conv_w[ki, c]
        conv_out[t, c] = s
print(f"\nStep 2: Conv1d")
print(f"  conv_out[0,0:5]: {conv_out[0,:5]}")

# ========== STEP 3: SiLU ==========
def silu(x):
    result = np.where(x < -80.0, 0.0, x / (1.0 + np.exp(-x)))
    return result

conv_silu = silu(conv_out)
print(f"\nStep 3: SiLU")
print(f"  conv_silu[0,0:5]: {conv_silu[0,:5]}")

# ========== STEP 4: Split Q, K, V ==========
# conv_silu: [T, 8192] = [T, Q(2048) + K(2048) + V(4096)]
q = conv_silu[:, :KEY_DIM]       # [T, 2048]
k = conv_silu[:, KEY_DIM:2*KEY_DIM]  # [T, 2048]
v = conv_silu[:, 2*KEY_DIM:]     # [T, 4096]
print(f"\nStep 4: Split Q/K/V")
print(f"  q[0,0:5]: {q[0,:5]}")
print(f"  k[0,0:5]: {k[0,:5]}")
print(f"  v[0,0:5]: {v[0,:5]}")

# ========== STEP 5: L2 Normalize Q and K ==========
# Q has shape [T, SSM_K_HEADS=16, SSM_D_STATE=128]
q_reshaped = q.reshape(T, SSM_K_HEADS, SSM_D_STATE)
k_reshaped = k.reshape(T, SSM_K_HEADS, SSM_D_STATE)
v_reshaped = v.reshape(T, SSM_V_HEADS, SSM_D_STATE)

q_norm = np.zeros_like(q_reshaped)
k_norm = np.zeros_like(k_reshaped)
for s in range(T):
    for h in range(SSM_K_HEADS):
        q_norm[s, h] = q_reshaped[s, h] / (np.linalg.norm(q_reshaped[s, h]) + 1e-12)
        k_norm[s, h] = k_reshaped[s, h] / (np.linalg.norm(k_reshaped[s, h]) + 1e-12)
print(f"\nStep 5: L2 Norm")
print(f"  q_norm[0,0,0:5]: {q_norm[0,0,:5]}")
print(f"  k_norm[0,0,0:5]: {k_norm[0,0,:5]}")

# ========== STEP 6: Head repeat (Q/K -> V heads) ==========
repeat_factor = SSM_V_HEADS // SSM_K_HEADS  # 2
q_repeated = np.repeat(q_norm, repeat_factor, axis=1)  # [T, 32, 128]
k_repeated = np.repeat(k_norm, repeat_factor, axis=1)  # [T, 32, 128]
print(f"\nStep 6: Head repeat")
print(f"  q_repeated shape: {q_repeated.shape}")

# ========== STEP 7: Gate projection ==========
z_gate = normed @ gate_w  # [T, 4096]
z_silu = silu(z_gate)  # [T, 4096]
print(f"\nStep 7: Gate projection")
print(f"  z_gate[0,0:5]: {z_gate[0,:5]}")
print(f"  z_silu[0,0:5]: {z_silu[0,:5]}")

# ========== STEP 8: Beta computation ==========
beta_raw = normed @ beta_w  # [T, DT_RANK]
beta = 1.0 / (1.0 + np.exp(-beta_raw))  # sigmoid. But beta has only DT_RANK=32 dims...
print(f"\nStep 8: Beta")
print(f"  beta_raw shape: {beta_raw.shape}")
print(f"  beta[0,0:5]: {beta[0,:5]}")

# Wait - beta has DT_RANK=32 elements per token, but we need 1 per V-head (32).
# In the code, beta_s[vh] indexes DT_RANK with vh... 
# So DT_RANK = SSM_V_HEADS = 32, no expansion needed. ✓

# ========== STEP 9: Gate (alpha + dt_bias) * ssm_a ==========
alpha_raw = normed @ alpha_w  # [T, DT_RANK]
alpha_biased = alpha_raw + dt_bias  # [T, 32]
# softplus
alpha_softplus = np.where(alpha_biased > 80, alpha_biased,
                          np.where(alpha_biased < -80, 0.0,
                                   np.log(1.0 + np.exp(alpha_biased))))
gate = alpha_softplus * ssm_a  # [T, 32]
gate_exp = np.exp(np.clip(gate, -80, 80))
print(f"\nStep 9: Gate")
print(f"  alpha_raw[0,0:5]: {alpha_raw[0,:5]}")
print(f"  alpha_biased[0,0:5]: {alpha_biased[0,:5]}")
print(f"  gate[0,0:5]: {gate[0,:5]}")
print(f"  exp(gate)[0,0:5]: {gate_exp[0,:5]}")

# ========== STEP 10: SSM Recurrence ==========
# State: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
q_scale = 1.0 / math.sqrt(SSM_D_STATE)
state = np.zeros((SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE))
output_all = np.zeros((T, VALUE_DIM))

for s in range(T):
    for vh in range(SSM_V_HEADS):
        kh = vh // repeat_factor
        bg = beta[s, vh]
        gg = gate_exp[s, vh]
        
        q_h = q_repeated[s, kh] * q_scale  # scaled Q
        k_h = k_repeated[s, kh]
        v_h = v_reshaped[s, vh]
        
        h = state[vh]  # [D_STATE, D_STATE]
        
        # State decay
        h *= gg
        
        # h @ k -> [D_STATE]
        hk = h @ k_h
        
        # diff = v - hk
        diff = v_h - hk
        
        # State update: h += k @ (diff * beta)
        h += np.outer(k_h, diff * bg)
        
        # Output: h @ q
        out = h @ q_h
        
        state[vh] = h
        output_all[s, vh * SSM_D_STATE:(vh+1) * SSM_D_STATE] = out

print(f"\nStep 10: SSM Recurrence")
print(f"  output_all[0,0:5]: {output_all[0,:5]}")
print(f"  output_all[0,120:130]: {output_all[0,120:130]}")

# ========== STEP 11: Gated Normalization ==========
output_normed = np.zeros((T, VALUE_DIM))
for s in range(T):
    for vh in range(SSM_V_HEADS):
        base = vh * SSM_D_STATE
        out_vh = output_all[s, base:base+SSM_D_STATE]
        z_vh = z_silu[s, base:base+SSM_D_STATE]
        
        sum_sq = np.sum(out_vh**2)
        rms = math.sqrt(sum_sq / SSM_D_STATE + 1e-6)
        scale = 1.0 / rms
        
        output_normed[s, base:base+SSM_D_STATE] = out_vh * scale * ssm_norm_w * z_vh

print(f"\nStep 11: Gated Normalization")
print(f"  output_normed[0,0:5]: {output_normed[0,:5]}")

# ========== STEP 12: Output Projection ==========
# output[D_MODEL] = ssm_out_w[T VALUE_DIM] -> dims: [VALUE_DIM, D_MODEL]
# In row-major: out[j] = sum_i inp[i] * w[i][j] where w[i][j] at offset i + j*VALUE_DIM
# In numpy with order='F': ssm_out_w[i, j] = data[i + j*VALUE_DIM]
final_out = output_normed @ ssm_out_w  # [T, D_MODEL]
print(f"\nStep 12: Output Projection")
print(f"  final_out[0,0:5]: {final_out[0,:5]}")
print(f"  final_out[0,0:5]: {final_out[0,:5]}")

# ========== COMPARE WITH C CODE OUTPUT ==========
# Our C code dumped the per-layer residual at /tmp/our_v2/layer_00.bin
# The residual includes: embd + SSM_output_proj + MoE_output
# But with MAX_LAYERS=1 the MoE is also computed.
# So final_out[0] = SSM contribution to token 0's residual
# full_residual = embd + SSM_contrib + MoE_contrib

our_layer0 = load('/tmp/our_v2/layer_00.bin').reshape(2, 2048)
ref_layer0 = load('/tmp/ref_dump/layer_00.bin').reshape(2, 2048)

# Initial embedding
init_emb = load('/tmp/our_initial_emb.bin').reshape(2, 2048)

# Check: does our layer0 residual = embd + SSM_contrib + MoE_contrib?
# layer0_residual = init_emb + final_out + ... (MoE contribution unknown)
# Actually our per-layer dump is AFTER MoE too. But we can check:
# expected = init_emb + final_out  (if no MoE)
# But there IS MoE with MAX_LAYERS=1

# Let me just compare the SSM contribution directly
# For BOS token: ref final = ref initial + SSM_ref + MoE_ref
# We don't have ref initial embedding separately
# But we can compute: SSM_ref_contrib = ref_layer0 - ref_initial_emb

# Actually, we don't have ref_initial_emb. But the ref_initial_emb should be
# the same as our initial emb (since we verified the embedding dequant is correct).

# Wait, but we DON'T have the reference's initial embedding. Let me check if
# ref_layer0 - our_layer0 matches ref_SSM_contrib - our_SSM_contrib

print(f"\n=== Comparison ===")
print(f"Initial emb norm: {np.linalg.norm(init_emb[0]):.4f}")
print(f"Our SSM contrib (final_out[0]) norm: {np.linalg.norm(final_out[0]):.4f}")
print(f"Expected residual (emb + SSM) norm: {np.linalg.norm(init_emb[0] + final_out[0]):.4f}")
print(f"Our actual layer0 residual[0] norm: {np.linalg.norm(our_layer0[0]):.4f}")
print(f"Ref layer0 residual[0] norm: {np.linalg.norm(ref_layer0[0]):.4f}")

# The difference is the MoE contribution
moe_our = our_layer0[0] - (init_emb[0] + final_out[0])
print(f"Our MoE contrib norm: {np.linalg.norm(moe_our):.4f}")

# Compare output_normed (VALUE_DIM) with our C code's DUMP_SSM_VAL output
ssm_val_our = load('/tmp/our_ssm_val.bin')  # from earlier dump
cs = np.dot(output_normed[0], ssm_val_our) / (np.linalg.norm(output_normed[0]) * np.linalg.norm(ssm_val_our) + 1e-30)
print(f"\nPython SSM vs C SSM (first token): cos_sim={cs:.6f}")
print(f"  Python norm: {np.linalg.norm(output_normed[0]):.4f}")
print(f"  C norm: {np.linalg.norm(ssm_val_our):.4f}")

# If they match, the SSM is correct!
