#!/usr/bin/env python3
"""Compare Q5_K dequant of attn_qkv between Python and C."""
import gguf
import numpy as np
import struct

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'blk.0.attn_qkv.weight'][0]
print(f"GGUF dims: {t.shape}")  # [2048, 8192]

# Get raw bytes, flattened
raw = np.array(t.data).view(np.uint8).flatten()
print(f"Raw bytes shape: {raw.shape}")

# First 2048 elements = 8 blocks of Q5_K (for outer=0)
D_MODEL = 2048
CONV_DIM = 8192

# Q5_K block: 176 bytes per 256 elements
QK_K = 256
BLOCK_SIZE = 176
# d(2) + dmin(2) + scales(12) + qh(6) + qs(128) = 150? No let me check
# Actually Q5_K: d(2) + dmin(2) + scales(6) + qh(4) + qs(128) = 142... 
# Let me just read the raw bytes and figure it out

# Function to dequant a single Q5_K block
def dequant_q5k_block(block_bytes):
    """block_bytes: 176 bytes. Return 256 floats."""
    d = struct.unpack('<h', block_bytes[0:2])[0]
    dmin = struct.unpack('<h', block_bytes[2:4])[0]
    # hmm, actually Q5_K uses f16 for d and dmin
    # Let me re-verify
    d = struct.unpack('<e', block_bytes[0:2])[0]  # f16
    dmin = struct.unpack('<e', block_bytes[2:4])[0]  # f16
    
    # Scales: 6 bytes + 6 bytes? No, Q5_K has 12 scale bytes
    # Actually let me check the real Q5_K block structure
    # The BLOCK_SIZE is 176 bytes:
    # d(2) + dmin(2) + scales(6) + qh(4) + qs(128) = 142? 
    # Or: d(2) + dmin(2) + scales(12) + qh(2) + qs(128) = 146?
    # Or: d(2) + dmin(2) + scales(6) + qh(4) + qs(128) = 142... hmm
    
    # Let me print the first block's structure
    return None

# Let me just read the raw data by examining the C output
c_data = np.fromfile('/tmp/c_attn_qkv_part.bin', dtype=np.float32)
print(f"C data: first 20: {[f'{x:.8f}' for x in c_data[:20]]}")
print(f"C data: mean={c_data.mean():.10f} std={c_data.std():.10f}")

# Now, let me try to use the gguf library's get() method
# For quantized tensors, we need to access the data field differently
# The t.data is a numpy array of (outer_dim, bytes_per_outer) = (8192, 1408)

data_shape = t.data.shape
print(f"data shape: {data_shape}")
# It should be (8192, 1408) for [2048, 8192] Q5_K since 8192*1408 bytes

# Let me try indexing t.data directly
# data[outer, byte_offset]
# For outer=0, byte 0: should be the first byte of d for the first block
if len(data_shape) == 2:
    n_outer, bytes_per_outer = data_shape
    print(f"n_outer={n_outer}, bytes_per_outer={bytes_per_outer}")
    
    # First block (inner 0..255, outer 0) starts at 
    outer = 0
    block_idx = 0  # first block within outer=0
    
    block_start = block_idx * BLOCK_SIZE
    block_bytes = bytes(raw[outer * bytes_per_outer + block_start : 
                             outer * bytes_per_outer + block_start + BLOCK_SIZE])
    
    print(f"First block bytes (first 20): {[hex(b) for b in block_bytes[:20]]}")
    
    d = struct.unpack('<e', block_bytes[0:2])[0]
    dmin = struct.unpack('<e', block_bytes[2:4])[0]
    print(f"d={d:.10f} dmin={dmin:.10f}")
    
    # Let me try other interpretations
    # Maybe it's different Q5_K variant
    # In llama.cpp, Q5_K is:
    # d: 2 bytes (half), dmin: 2 bytes (half)
    # scales: 6 bytes
    # qh: 4 bytes
    # qs: 128 bytes
    # Total: 2+2+6+4+128 = 142... but block size is 176
    # OH WAIT, the block size includes:
    # d(2), dmin(2), block_scale_0(4), block_scale_1(4), block_scale_2(4), qh(6), qs(128) = 150?
    # Let me try 142
    
    # Actually in llama.cpp Q5_K_BLOCK_SIZE is defined somewhere
    # Let me check by trying to read the raw file
    
    # Try: d(2)+dmin(2)+scales(12)+qh(4)+qs(128) = 148... still not 176
    
    # Try: d(2)+dmin(2)+scales(12)+qh(12)+qs(128) = 156
    # Try: d(2)+dmin(2)+scales(6)+qh(6)+qs(128) = 144
    
    # Hmm, let me just print the full block to understand
    print(f"Full block hex: {block_bytes.hex()}")
