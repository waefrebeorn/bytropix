#!/usr/bin/env python3
"""Verify Q6_K dequant: ggml block layout, compare with C."""
import struct
import numpy as np
import gguf

# block_q6_K from ggml-common.h:
#   ql[128] (offset 0)  - low 4 bits
#   qh[64]  (offset 128) - high 2 bits
#   scales[16] (offset 192) - int8
#   d (offset 208) - ggml_half/f16
# Total: 210 bytes

# Load raw ssm_out bytes from gguf
r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'blk.0.ssm_out.weight'][0]
raw = np.array(t.data).view(np.uint8).flatten()

# First Q6_K block
block0 = bytes(raw[0:210])
print(f"Block size: {len(block0)} bytes")

# Parse using CORRECT layout
d = struct.unpack('<e', block0[208:210])[0]
ql = list(block0[0:128])
qh = list(block0[128:192])
sc = list(struct.unpack('16b', block0[192:208]))  # 16 signed int8

print(f"d={d:.8f}")
print(f"scales[0..3]: {sc[:4]}")
print(f"ql[0..3]: {ql[:4]}")
print(f"qh[0..3]: {qh[:4]}")

# Dequant matching reference dequantize_row_q6_K
result = np.zeros(256, dtype=np.float32)

for n in range(0, 256, 128):  # 2 iterations
    ql_off = n // 2  # advance 64
    qh_off = n // 4  # advance 32
    sc_off = 0 if n == 0 else 8  # advance 8
    
    for l in range(32):
        is_idx = l // 16
        
        q1 = (ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)
        q2 = (ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)
        q3 = (ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)
        q4 = (ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)
        
        q1 -= 32; q2 -= 32; q3 -= 32; q4 -= 32
        
        result[n + l + 0] = float(d) * sc[sc_off + is_idx + 0] * q1
        result[n + l + 32] = float(d) * sc[sc_off + is_idx + 2] * q2
        result[n + l + 64] = float(d) * sc[sc_off + is_idx + 4] * q3
        result[n + l + 96] = float(d) * sc[sc_off + is_idx + 6] * q4

print(f"\nFirst 10: {[f'{x:.8f}' for x in result[:10]]}")
print(f"Mean={result.mean():.8f} Std={result.std():.8f}")

# Compare with C
c_ssm_out = np.fromfile('/tmp/c_ssm_out_w.bin', dtype=np.float32)
c_block0 = c_ssm_out[:256]
diff = np.abs(c_block0 - result)
print(f"\nComparison with C (first 256 elements):")
print(f"  Max diff: {diff.max():.10f}")
print(f"  C first 10: {[f'{x:.8f}' for x in c_block0[:10]]}")
if diff.max() < 1e-5:
    print(" ✅ Q6_K DEQUANT MATCHES!")
else:
    print(" ❌ Q6_K DEQUANT MISMATCH!")
    bad = np.where(diff > 1e-5)[0]
    print(f"  {len(bad)} mismatching elements")
    for i in bad[:5]:
        print(f"  [{i}]: C={c_block0[i]:.10f} Py={result[i]:.10f} diff={diff[i]:.10f}")
