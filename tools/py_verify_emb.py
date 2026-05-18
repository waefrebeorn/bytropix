#!/usr/bin/env python3
"""Verify embedding extraction by comparing against gguf's token_embd."""
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]

# t.data shape: (248320, 1408) - raw bytes for Q5_K
# But for Python gguf library, we need to handle this differently.
# Actually, the t.data in gguf Python library might already be dequantized for some versions.
# Let me check what dtype it has.

print(f"tensor data type: {t.data.dtype}")
print(f"tensor data shape: {t.data.shape}")
print(f"tensor type: {gguf.GGMLQuantizationType(t.tensor_type)}")

# If it's raw bytes (uint8), we need to dequant
# If it's float32, it's already dequantized
if t.data.dtype == np.float32:
    # Already dequantized
    print("Data is float32 - already dequantized")
    data = np.array(t.data)
    # shape: (248320, 2048) or similar
    # For Q5_K: each 256-element block = 176 bytes
    # With 2048 dims = 8 blocks per token
    # raw bytes shape: (248320, 1408) = (248320, 8*176)
    # After dequant: (248320, 2048)
    print(f"Data shape: {data.shape}")
    emb_bos = data[248044]  # row 248044 for BOS token
elif t.data.dtype == np.uint8:
    # Raw bytes - need to dequant
    print("Data is uint8 - raw bytes, need dequant")
    raw = np.array(t.data)  # (248320, 1408)
    print(f"Raw shape: {raw.shape}")
    
    # Read our C-extracted embedding
    c_emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32,
                        count=2048, offset=248044*2048*4)
    print(f"C embedding first 10: {[f'{x:.8f}' for x in c_emb[:10]]}")
    print(f"C embedding: mean={c_emb.mean():.8f} std={c_emb.std():.8f}")
    
    # Read token 0 from C file
    c_tok0 = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32,
                         count=2048, offset=0)
    print(f"C token 0 first 10: {[f'{x:.8f}' for x in c_tok0[:10]]}")
    print(f"C token 0: mean={c_tok0.mean():.8f} std={c_tok0.std():.8f}")
    
    # Also try reading token 248044 from C but verify against known values
    # Read token 1 from C file
    c_tok1 = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32,
                         count=2048, offset=1*2048*4)
    print(f"\nC token 1 first 10: {[f'{x:.8f}' for x in c_tok1[:10]]}")
    print(f"C token 1: mean={c_tok1.mean():.8f} std={c_tok1.std():.8f}")
    
    # Let me also check the exact raw bytes for token 248044
    # In our file, token 248044 starts at byte 248044*2048*4
    with open('data/qwen36_embeddings_c.bin.raw', 'rb') as f:
        f.seek(248044*2048*4)
        raw_c = f.read(20*4)  # first 20 floats
    import struct
    vals = struct.unpack(f'<{20}f', raw_c)
    print(f"\nC raw bytes -> floats first 20: {[f'{x:.8f}' for x in vals]}")

else:
    print(f"Unexpected dtype: {t.data.dtype}")
