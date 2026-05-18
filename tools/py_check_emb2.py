#!/usr/bin/env python3
"""Get proper dequantized BOS embedding from GGUF."""
import numpy as np
import gguf

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]

# GGUFReader returns quantized data as uint8 view
# We need to extract the raw bytes for the specific token
# t.data shape = [dims...], it's a numpy array of quantized data
print(f"Tensor shape: {t.shape}")
print(f"Tensor type code: {int(t.tensor_type)}")

# The actual data is a 2D array if accessed differently
# Let's check the raw byte layout
raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
print(f"Raw bytes: {len(raw)}")

# For Q5_K, each block is 256 elements packed into 144 bytes
# BOS token ID = 248044
# Block index for this token
QK_K = 256
block_idx = 248044 // QK_K
elem_in_block = 248044 % QK_K
print(f"BOS token: block={block_idx}, elem_in_block={elem_in_block}")

# Use gguf tensor data properly - t[0] returns a view of first row
# For 2D quantized tensor, we need to dequantize manually or use library
# Let's try accessing via the tensor's stored data
if hasattr(t.data, 'shape'):
    print(f"t.data shape (all dims): {t.data.shape}")
    if len(t.data.shape) == 1:
        # Flat array - reshape based on tensor shape info
        print(f"Flat data length: {t.data.shape[0]}")
else:
    print(f"t.data type: {type(t.data)}")

# Use numpy to read the embedding file that we extracted
our_emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32)
# reshape to [n_tokens, dim]
our_emb = our_emb.reshape(-1, 2048)
print(f"Our embedding file: shape={our_emb.shape}, tokens={our_emb.shape[0]}")

bos = our_emb[248044]
print(f"\nOur BOS embedding:")
print(f"  mean={bos.mean():.6f}, std={bos.std():.6f}")
print(f"  [0..4]: {[f'{x:.8f}' for x in bos[:5]]}")

# Now load reference final hidden
ref = np.fromfile('/tmp/llama_embd.bin', dtype=np.float32)
print(f"\nReference final hidden (2048 dim):")
print(f"  mean={ref.mean():.6f}, std={ref.std():.6f}")
print(f"  [0..4]: {[f'{x:.8f}' for x in ref[:5]]}")

# Load our final hidden
our = np.fromfile('/tmp/our_hidden.bin', dtype=np.float32)
print(f"\nOur final hidden:")
print(f"  mean={our.mean():.6f}, std={our.std():.6f}")
print(f"  [0..4]: {[f'{x:.8f}' for x in our[:5]]}")

# Compare
cos = np.dot(our, ref) / (np.linalg.norm(our) * np.linalg.norm(ref))
print(f"\nCos-sim: {cos:.6f}")

# Check the layer 0 output
l0_in = np.fromfile('/tmp/dump_layers/our_layer_0_in.bin', dtype=np.float32)
l0_out = np.fromfile('/tmp/dump_layers/our_layer_0_out.bin', dtype=np.float32)

# l0_in should be the BOS embedding
diff = np.abs(l0_in - bos)
print(f"\nLayer 0 input vs BOS embedding:")
print(f"  Max diff: {diff.max():.10f}")
if diff.max() < 1e-5:
    print("  ✅ EXACT MATCH (layer 0 input = BOS embedding)")
else:
    print(f"  ❌ MISMATCH")
    print(f"  l0_in first: {[f'{x:.6f}' for x in l0_in[:5]]}")
    print(f"  bos  first: {[f'{x:.6f}' for x in bos[:5]]}")
