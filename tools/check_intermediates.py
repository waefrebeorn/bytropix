#!/usr/bin/env python3
"""Dequant token_embd and SSM weights from GGUF raw data,
then implement the SSM forward pass to compare with C output."""
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

# Constants
D_MODEL = 2048
SSM_K_HEADS = 16
SSM_V_HEADS = 32
SSM_D_STATE = 128
KEY_DIM = SSM_D_STATE * SSM_K_HEADS  # 2048
VALUE_DIM = SSM_D_STATE * SSM_V_HEADS  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 8192
CONV_KERNEL = 4
DT_RANK = 32
QK_K = 256
Q5_K_BLOCK_SIZE = 176
Q6_K_BLOCK_SIZE = 164  # 2+2+12+128+? Actually Q6_K: d(2)+dmin(2)+scales(12)+qh(0)+qs(128)+ql(128)? No
# Q6_K: d(2) + dmin(2) + scales(12) + ql(128) + qh(128) = 276... no
# Actually let me just check
# Q6_K: d(2)+dmin(2)+scales(12)+qs(256) = 272 bytes per 256 elements? 
# Hmm, I need to check the actual block size

# Let me just read from the saved C intermediates instead
# C already saved: /tmp/c_emb.bin, /tmp/c_normed.bin, /tmp/c_ssm_out.bin

c_emb = np.fromfile('/tmp/c_emb.bin', dtype=np.float32)
c_normed = np.fromfile('/tmp/c_normed.bin', dtype=np.float32)
c_ssm_out = np.fromfile('/tmp/c_ssm_out.bin', dtype=np.float32)

print(f"C emb: mean={c_emb.mean():.8f} std={c_emb.std():.8f}")
print(f"C normed: mean={c_normed.mean():.8f} std={c_normed.std():.8f}")
print(f"C SSM out: mean={c_ssm_out.mean():.8f} std={c_ssm_out.std():.8f}")
print(f"C SSM out first 10: {[f'{x:.8f}' for x in c_ssm_out[:10]]}")
