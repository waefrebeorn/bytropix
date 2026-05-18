#!/usr/bin/env python3
"""Read ALL weights one by one from GGUF using gguf library, 
then run the SSM forward pass for layer 0 manually.
Compare every intermediate step against our C code dumps."""
import gguf
import numpy as np
import struct
import sys

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

def dequant_Q5_K(block, d, dmin, out, base, D):
    """Dequant a single Q5_K block (256 elements, 176 bytes)"""
    # Block layout: d(2) + dmin(2) + scales(6) + qh(8) + qs(128) = 146... wait
    # Actually Q5_K: d(2) + dmin(2) + scales(6) + qh(4) + qs(128) = 142??
    # Let me just read the block size and figure out from there
    pass

# Let me try the simplest possible thing: 
# Read ssm_beta_weight (F32) and compare with our C code
ssm_beta = np.array([t for t in r.tensors if t.name == 'blk.0.ssm_beta.weight'][0].data)
ssm_beta = np.array(ssm_beta, dtype=np.float32).reshape(2048, 32)
# In Python gguf, shape is (32, 2048) since dims=[2048,32]
print(f"ssm_beta shape: {ssm_beta.shape}")

# Read from our embedding file
emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32, count=2048, offset=248044*2048*4)
print(f"emb shape: {emb.shape} mean={emb.mean():.8f} std={emb.std():.8f}")

# Compute beta raw
beta_raw = emb @ ssm_beta.T  # [2048] @ [32, 2048] -> [32]
print(f"beta_raw: {beta_raw}")

# Read attn_norm weight (F32)
attn_norm = np.array([t for t in r.tensors if t.name == 'blk.0.attn_norm.weight'][0].data, dtype=np.float32)
print(f"attn_norm shape: {attn_norm.shape}")
