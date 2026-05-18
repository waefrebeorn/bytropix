#!/usr/bin/env python3
"""Understand the gguf library data layout by comparing against raw bytes."""
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')

# ssm_beta is F32, no dequant needed. Dims [2048, 32].
t = [t for t in r.tensors if t.name == 'blk.0.ssm_beta.weight'][0]
print(f"GGUF dims: {t.shape}")
print(f"NUMPY shape: {t.data.shape}")
print(f"dtype: {t.data.dtype}")

# Let's look at the actual layout
data = np.array(t.data)
print(f"\nFirst 5 values: {data[0, :5]}")  # numpy[0, 0:5]
print(f"data[0,0] = {data[0, 0]:.8f}")
print(f"data[1,0] = {data[1, 0]:.8f}")

# Now read the raw bytes from the file to verify
# For F32: 2048 * 32 * 4 = 262,144 bytes
# The file position depends on how gguf structures tensors
# Let me find the offset
offset = t.field.offset
print(f"GGUF offset: {offset}")

with open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb') as f:
    f.seek(offset)
    raw = f.read(2048 * 32 * 4)  # 2048*32 = 65536 floats = 262,144 bytes

raw_floats = np.frombuffer(raw, dtype=np.float32)
print(f"Raw first 5: {raw_floats[:5]}")
print(f"Raw[0] = {raw_floats[0]:.8f} raw[2048] = {raw_floats[2048]:.8f}")

# Compare: numpy[0,0] should = raw_float[0] (or raw_float[0*32+0] if col-major, or raw_float[0] if inner=2048)
# numpy[1,0] should = raw_float[?]
print(f"\nRaw[0] vs data[0,0]: {raw_floats[0]:.8f} vs {data[0, 0]:.8f}")
print(f"Raw[32] vs data[1,0]: {raw_floats[32]:.8f} vs {data[1, 0]:.8f}")
print(f"Raw[2048] vs data[0,1]: {raw_floats[2048]:.8f} vs data[?,?]")

# So for gguf dims [2048, 32] (inner=2048, outer=32):
# element (outer_j, inner_i) at offset inner_i + outer_j * 2048
# numpy shape (2048, 32) with numpy[r, c] at offset r*32 + c
# numpy[r, c] corresponds to GGUF element with inner=r, outer=c IF 
#   data is NOT transposed and axes are same
# r*32 + c vs inner_i + outer_j * 2048
# For i=0, j=0: both = 0 ✓
# For i=1, j=0: numpy offset = 1*32+0=32, GGUF offset = 0+0*2048=0 ✗

print(f"\nGGUF offset system (inner=dim0=2048, outer=dim1=32):")
print(f"  element(inner=0, outer=0) = raw[0] = {raw_floats[0]:.8f}")
print(f"  element(inner=1, outer=0) = raw[1] = {raw_floats[1]:.8f}")
print(f"  element(inner=0, outer=1) = raw[2048] = {raw_floats[2048]:.8f}")

# In numpy (2048, 32): axis 0 = 2048, axis 1 = 32
# numpy[i, j] = raw[i * 32 + j]
# This maps to GGUF element(inner=j, outer=i) since raw[i*32 + j] = inner=j + outer=i*2048 only if i*32 + j = j + i*2048
# which requires 32 = 2048. So the LAYOUT IS DIFFERENT!

print(f"\nnumpy[0,0] = raw[{0*32+0}] = {raw_floats[0*32+0]:.8f}")
print(f"numpy[1,0] = raw[{1*32+0}] = {raw_floats[1*32+0]:.8f}")
print(f"numpy[0,1] = raw[{0*32+1}] = {raw_floats[0*32+1]:.8f}")

# So in numpy: [i, j] = raw[i * 32 + j]
# For this to match GGUF (inner=i, outer=j): i * 32 + j should equal inner_i + outer_j * 2048
# i*32+j = i + j*2048 → only if i*(32-1) = j*(2048-1) which isn't generally true
# The numpy and raw are just DIFFERENT layouts

# Actually, the gguf library might NOT transpose. It might just wrap the raw data
# with the GGUF dims interpreted as numpy axes.
# For dims [2048, 32], numpy shape = (2048, 32).
# numpy[i, j] = raw[i * 32 + j]
# But GGUF stores as: first 2048 values for outer=0, next 2048 for outer=1, etc.
# raw[outer_j * 2048 + inner_i] 
# For GGUF element (inner=i, outer=j): raw[j * 2048 + i]

# Using numpy's indexing on this raw data: 
# To get element (inner=i, outer=j): data[j, i] would be at raw[j * 32 + i] (not raw[j*2048+i])

# THIS IS WRONG! The numpy wrapper doesn't match the GGUF layout!
# The gguf library creates: data = raw.reshape(dims[1], dims[0]) 
# So data[j][i] = raw[j * dims[0] + i] = raw[j * 2048 + i] = GGUF element (inner=i, outer=j)

# Let me verify this hypothesis
print(f"\nTesting gguf reshape hypothesis:")
print(f"  data[0,1] = {data[0, 1]:.8f} vs raw[1] = {raw_floats[1]:.8f}")
print(f"  data[1,0] = {data[1, 0]:.8f} vs raw[2048] = {raw_floats[2048]:.8f}")

# If data[j, i] = raw[j * 2048 + i]:
# data[0, 1] = raw[0 * 2048 + 1] = raw[1] ✓
# data[1, 0] = raw[1 * 2048 + 0] = raw[2048]
# But this gives: data[0, 1] = raw[1], data shape is (2048, 32)
# data[1, 0] = raw[2048], that would mean data[1,0] should equal raw[2048]
# Let me check!
print(f"  data[1,0] vs raw[2048]: {data[1, 0]:.8f} vs {raw_floats[2048]:.8f}")
