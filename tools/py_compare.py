#!/usr/bin/env python3
"""Python reference: verify SSM layer 0 against C dumps.
GGUF library transposes: for dims [2048, 32], numpy shape is (32, 2048).
data[outer, inner] maps to our C code's weight[i + outer * D_MODEL]."""
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

D_MODEL = 2048
VOCAB = 248320
SSM_K_HEADS = 16
SSM_V_HEADS = 32
SSM_D_STATE = 128
KEY_DIM = SSM_D_STATE * SSM_K_HEADS  # 2048
VALUE_DIM = SSM_D_STATE * SSM_V_HEADS  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 8192
CONV_KERNEL = 4
DT_RANK = 32

def load(name):
    """Load F32 tensor from gguf, return numpy array."""
    t = [t for t in r.tensors if t.name == name][0]
    return np.array(t.data, dtype=np.float32)

# Load weights (all F32)
attn_norm = load('blk.0.attn_norm.weight')  # (2048,)
ssm_beta = load('blk.0.ssm_beta.weight')      # (32, 2048)
ssm_alpha = load('blk.0.ssm_alpha.weight')     # (32, 2048)
ssm_dt = load('blk.0.ssm_dt.bias')             # (32,)
ssm_a = load('blk.0.ssm_a')                     # (32,)

# ssm_conv1d: GGUF dims [4, 8192] -> numpy shape (8192, 4)
ssm_conv1d = load('blk.0.ssm_conv1d.weight')   # (8192, 4)
ssm_norm = load('blk.0.ssm_norm.weight')        # (128,)
# ssm_out: GGUF dims [4096, 2048] -> numpy shape (2048, 4096)
ssm_out = load('blk.0.ssm_out.weight')          # (2048, 4096)

# Load quantized weights (use our C code's dequant via embedding file extraction)
# For attn_qkv and attn_gate (Q5_K), we need to dequant manually OR use C helper
# Let me save these from our C program instead

# Load embedding from C-extracted file
emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32,
                   count=D_MODEL, offset=248044*D_MODEL*4)
print(f"Embedding: mean={emb.mean():.8f} std={emb.std():.8f}")

# Also load C's normed output  
c_normed = np.fromfile('/tmp/c_normed.bin', dtype=np.float32)
print(f"C normed: mean={c_normed.mean():.8f} std={c_normed.std():.8f}")

# Step 0: RMSNorm
def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return x / rms * w

py_normed = rms_norm(emb, attn_norm)
print(f"Py normed: mean={py_normed.mean():.8f} std={py_normed.std():.8f}")
print(f"Match? maxdiff={np.max(np.abs(py_normed - c_normed)):.10f}")

# Step 1: Beta and Alpha projections
# ssm_beta: (32, 2048) -> data[vh, i] = weight[i + vh * D_MODEL]
# beta_raw[vh] = sum_i x[i] * weight[i + vh * D_MODEL] = sum_i x[i] * data[vh, i]
beta_raw = emb @ ssm_beta.T  # [2048] @ [2048, 32] = [32]
alpha_raw = emb @ ssm_alpha.T
print(f"\nBeta raw: {[f'{x:.8f}' for x in beta_raw]}")
print(f"Alpha raw: {[f'{x:.8f}' for x in alpha_raw]}")

# Compute beta and gate
def softplus(x):
    x = np.clip(x, -80, 80)
    return np.log(1.0 + np.exp(x))

beta = 1.0 / (1.0 + np.exp(-beta_raw))
alpha_bias = alpha_raw + ssm_dt
alpha_sp = softplus(alpha_bias)
gate = alpha_sp * ssm_a
print(f"Beta: {[f'{x:.8f}' for x in beta]}")
print(f"Gate (exp part): {[f'{x:.8f}' for x in gate]}")
print(f"exp(gate): {[f'{x:.8f}' for x in np.exp(gate)]}")

# Step 5: Conv1d
# We need qkv_all (attn_qkv * input) which uses Q5_K dequant
# Let me load that from C dumps instead
# For now, let me just read the qkv output from the C dump 
# by running dump_intermediates with DUMP_SSM_DEBUG set
print("\n--- For full QKV comparison, need Q5_K dequant ---")
print("Let me use the C dumps instead")
