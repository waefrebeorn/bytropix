#!/usr/bin/env python3
"""Verify Q4_K output weight columns between Python and C.
Q4_K block from ggml: 256 elements, 144 bytes
  d: f16 (2 bytes) - super-block scale
  dmin: f16 (2 bytes) - super-block min
  scales: 12 bytes (6 low + 6 high, encodes 8 scale+min pairs via get_scale_min_k4)
  qs: 128 bytes (256 low/high nibbles)
"""
import struct
import numpy as np

def get_scale_min_k4(j, sh):
    if j < 4:
        d_sc = sh[j] & 63
        m_sc = sh[j + 4] & 63
    else:
        d_sc = (sh[j+4] & 0xF) | ((sh[j-4] >> 6) << 4)
        m_sc = (sh[j+4] >> 4) | ((sh[j-0] >> 6) << 4)
    return d_sc, m_sc

def dequant_q4k_block(block_bytes):
    """Dequant a Q4_K block. block_bytes: 144 bytes. Returns 256 floats."""
    d = struct.unpack('<e', block_bytes[0:2])[0]
    dmin = struct.unpack('<e', block_bytes[2:4])[0]
    scales = list(block_bytes[4:16])  # 12 bytes
    qs = block_bytes[16:144]  # 128 bytes
    
    result = np.zeros(256, dtype=np.float32)
    
    is_idx = 0
    for j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_idx, scales)
        sc2, m2 = get_scale_min_k4(is_idx+1, scales)
        d1 = float(d) * sc
        m1v = float(dmin) * m
        d2 = float(d) * sc2
        m2v = float(dmin) * m2
        
        qsg = qs[is_idx//2 * 32 : (is_idx//2 + 1) * 32]
        for l in range(32):
            idx = j + l
            result[idx] = d1 * (qsg[l] & 0xF) - m1v
            idx = j + 32 + l
            result[idx] = d2 * (qsg[l] >> 4) - m2v
        is_idx += 2
    
    return result

# Load raw bytes from GGUF for output.weight
import gguf
r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'output.weight'][0]
raw = np.array(t.data).view(np.uint8).flatten()

print(f"output.weight: shape={t.shape}, raw bytes={len(raw)}")
# Q4_K: 256 elem per block -> 144 bytes per block
# output.weight: [2048, 248320] -> total 248320 * 2048 / 256 * 144 = 248320 * 8 * 144 = 286,064,640 bytes
# raw should be (286064640,) or similar

# Dequant column 0 (outer=0): blocks at rows 0..7, each at raw[outer * 8 * 144 + block * 144 : ...]
# Wait, in the gguf Python library, the raw data for output.weight is 
# [dims[1], bytes_per_outer] = [248320, 1152] where 1152 = 8*144
# Let me check the data shape
data_shape = t.data.shape
print(f"data shape: {data_shape}")

n_outer = data_shape[0]  # 248320
bytes_per_outer = data_shape[1]  # 1152 = 8*144

# Read column 0: first 8 blocks
col0_py = np.zeros(2048, dtype=np.float32)
for block_idx in range(8):
    block_start = block_idx * 144
    block_end = block_start + 144
    block_bytes = bytes(raw[block_start:block_end])
    deq = dequant_q4k_block(block_bytes)
    col0_py[block_idx*256:(block_idx+1)*256] = deq

# Read column 220: starts at outer=220 * 1152 bytes
col220_py = np.zeros(2048, dtype=np.float32)
start = 220 * 1152
for block_idx in range(8):
    block_start = start + block_idx * 144
    block_end = block_start + 144
    block_bytes = bytes(raw[block_start:block_end])
    deq = dequant_q4k_block(block_bytes)
    col220_py[block_idx*256:(block_idx+1)*256] = deq

# Compare with C values
c_col0 = np.fromfile('/tmp/c_outw_col0.bin', dtype=np.float32)
c_col220 = np.fromfile('/tmp/c_outw_col220.bin', dtype=np.float32)

diff0 = np.abs(col0_py - c_col0)
diff220 = np.abs(col220_py - c_col220)

print(f"Column 0: maxdiff={diff0.max():.10f} mean={col0_py.mean():.8f} std={col0_py.std():.8f}")
print(f"  C: mean={c_col0.mean():.8f} std={c_col0.std():.8f}")
print(f"  First 5 C: {[f'{x:.8f}' for x in c_col0[:5]]}")
print(f"  First 5 Py: {[f'{x:.8f}' for x in col0_py[:5]]}")

print(f"\nColumn 220: maxdiff={diff220.max():.10f}")
print(f"  First 5 C: {[f'{x:.8f}' for x in c_col220[:5]]}")
print(f"  First 5 Py: {[f'{x:.8f}' for x in col220_py[:5]]}")

if diff0.max() > 1e-6:
    print("\n✗ Q4_K DEQUANT MISMATCH at column 0!")
    bad = np.where(diff0 > 1e-6)[0]
    print(f"  {len(bad)} mismatching elements")
    for i in bad[:5]:
        print(f"  [{i}]: C={c_col0[i]:.10f} Py={col0_py[i]:.10f} diff={diff0[i]:.10f}")
elif diff220.max() > 1e-6:
    print("\n✗ Q4_K DEQUANT MISMATCH at column 220!")
else:
    print("\n✓ Q4_K DEQUANT MATCHES for both columns!")
    
# Also read column 84944
col84944_py = np.zeros(2048, dtype=np.float32)
start = 84944 * 1152
for block_idx in range(8):
    block_start = start + block_idx * 144
    block_end = block_start + 144
    block_bytes = bytes(raw[block_start:block_end])
    deq = dequant_q4k_block(block_bytes)
    col84944_py[block_idx*256:(block_idx+1)*256] = deq

c_col84944 = np.fromfile('/tmp/c_outw_col84944.bin', dtype=np.float32)
diff84944 = np.abs(col84944_py - c_col84944)
print(f"\nColumn 84944: maxdiff={diff84944.max():.10f}")
if diff84944.max() > 1e-6:
    print("✗ Q4_K DEQUANT MISMATCH at column 84944!")
