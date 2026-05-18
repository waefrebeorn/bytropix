#!/usr/bin/env python3
"""Full Python reference: Q5_K dequant, compare against C values."""
import struct
import numpy as np

# ========== Q5_K Dequant ==========
def get_scale_min_k4(j, q):
    """Extract 6-bit scale and min from q[12] at position j (0-7)."""
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
    return d, m

def dequant_q5k_block(block_bytes):
    """Dequant Q5_K: 176 bytes -> 256 floats. Matches ggml-quants.c dequantize_row_q5_K."""
    assert len(block_bytes) == 176
    d = struct.unpack('<e', block_bytes[0:2])[0]
    dmin = struct.unpack('<e', block_bytes[2:4])[0]
    scales = list(block_bytes[4:16])  # 12 bytes
    qh = list(block_bytes[16:48])     # 32 bytes = 256 bits
    qs = list(block_bytes[48:176])    # 128 bytes = 256 low nibbles
    
    result = np.zeros(256, dtype=np.float32)
    
    is_idx = 0
    u1, u2 = 1, 2
    for j in range(0, 256, 64):  # 4 iterations
        sc, m = get_scale_min_k4(is_idx + 0, scales)
        d1 = float(d) * sc; m1 = float(dmin) * m
        sc, m = get_scale_min_k4(is_idx + 1, scales)
        d2 = float(d) * sc; m2v = float(dmin) * m
        
        ql = qs[is_idx//2 * 32 : (is_idx//2 + 1) * 32]
        
        for l in range(32):
            result[j + l] = d1 * ((ql[l] & 0xF) + (16 if (qh[l] & u1) else 0)) - m1
            result[j + 32 + l] = d2 * ((ql[l] >> 4) + (16 if (qh[l] & u2) else 0)) - m2v
        
        is_idx += 2
        u1 <<= 2
        u2 <<= 2
    
    return result

# ========== Test ==========
import gguf
r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

# Get attn_qkv raw bytes [shape=2048,8192 Q5_K]
t = [t for t in r.tensors if t.name == 'blk.0.attn_qkv.weight'][0]
raw = np.array(t.data).view(np.uint8).flatten()
data_shape = t.data.shape

# First block (inner 0..255, outer 0)
block0 = bytes(raw[0:176])
deq = dequant_q5k_block(block0)

print(f"Q5_K dequant (first block attn_qkv):")
print(f"  d={struct.unpack('<e', block0[0:2])[0]:.8f}")
print(f"  dmin={struct.unpack('<e', block0[2:4])[0]:.8f}")
print(f"  First 10: {[f'{x:.8f}' for x in deq[:10]]}")
print(f"  Mean={deq.mean():.8f} Std={deq.std():.8f}")

# Compare with C dequant
c_attn_qkv = np.fromfile('/tmp/c_attn_qkv_part.bin', dtype=np.float32, count=32768)
c_block0 = c_attn_qkv[:256]
diff = np.abs(c_block0 - deq)
print(f"\nComparison with C:")
print(f"  Max diff: {diff.max():.10f}")
print(f"  C first 10: {[f'{x:.8f}' for x in c_block0[:10]]}")

if diff.max() < 1e-5:
    print(" ✅ Q5_K DEQUANT MATCHES!")
else:
    print(" ❌ Q5_K DEQUANT MISMATCH!")
    bad = np.where(diff > 1e-5)[0]
    print(f"  {len(bad)} mismatching elements")
    for i in bad[:5]:
        print(f"  [{i}]: C={c_block0[i]:.10f} Py={deq[i]:.10f} diff={diff[i]:.10f}")
