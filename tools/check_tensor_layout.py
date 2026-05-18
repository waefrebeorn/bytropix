#!/usr/bin/env python3
"""Check the actual memory layout of 3D MoE tensors in the GGUF file.

For a 3D tensor with dims=[2048, 512, 256] (D_MODEL, D_FF, N_EXPERTS),
the GGUF stores data in row-major (C-order): last dimension varies fastest.

This means for each (k, j) pair where k=D_MODEL idx, j=D_FF idx,
all 256 expert values are contiguous.

But for quantized types like IQ2_XXS, each block covers 256 elements.
Since N_EXPERTS=256 is the fastest-varying dimension, one IQ2_XXS block 
covers exactly 1 (k,j) position × 256 experts.

Let's verify this by checking the raw data at known offsets.
"""
import struct
import numpy as np

f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4)  # magic
ver = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]

print(f"GGUF version: {ver}, tensors: {tensor_count}, KV: {kv_count}")

# Skip KV pairs
for i in range(kv_count):
    n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    typ = struct.unpack('<i', f.read(4))[0]
    if typ == 8:
        n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif typ == 9:
        arr_type = struct.unpack('<i', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            if arr_type == 8: n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
            else: f.read(4)
    elif typ in (10, 11, 12): f.read(8)
    elif typ == 6: f.read(4)
    elif typ == 7: f.read(1)
    elif typ in (0, 1): f.read(1)
    elif typ in (2, 3): f.read(2)
    elif typ in (4, 5): f.read(4)

# Read tensor info
tensors = []
for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims, ggml_type, data_offset))

# Get the gate_exps tensor info for blk.0
target_tensors = [t for t in tensors if 'blk.0.ffn_gate_exps' in t[0] or 
                  'blk.0.ffn_gate_inp' in t[0] or
                  'blk.0.ffn_down_exps' in t[0]]

for name, nd, dims, gtype, offset in target_tensors:
    print(f"\n{name}:")
    print(f"  dims = {dims}")
    print(f"  type = {gtype}")
    print(f"  data_offset = {offset}")
    
    # For F32 gate_inp, check the layout
    if gtype == 0:  # F32
        # Read the first 256*4 bytes = first row? Or first column?
        f.seek(offset)
        data = struct.unpack(f'<{min(256*4, 2048)}f', f.read(min(256*4, 2048)*4))
        print(f"  First 16 floats: {data[:16]}")
        print(f"  First row (e=0..255): same as above?")
        # If dims = [2048, 256] with C-order (last dim fastest):
        # ne[0]=256, ne[1]=2048
        # data[k * 256 + e] = value for row k, col e
        # First 256 values = k=0, e=0..255 = router weights connecting input dim 0 to all experts
        print(f"  These values represent router weight[dim=0, expert=0..255]")
    
    # For IQ2_XXS gate_exps, check the block structure
    if gtype == 16:  # IQ2_XXS
        # dims = [2048, 512, 256]
        # C-order: varies as (k, j, e) where k=D_MODEL, j=D_FF, e=N_EXPERTS
        # Each IQ2_XXS block = 66 bytes covering 256 elements
        # Since N_EXPERTS=256 is fastest, one block = 1 (k,j) × all 256 experts
        
        # Read first block (66 bytes)
        f.seek(offset)
        raw = f.read(66)
        print(f"  First 66 bytes of raw IQ2_XXS data:")
        print(f"    d (fp16): {raw[0]:02x} {raw[1]:02x}")
        print(f"    First 16 qs bytes: {' '.join(f'{b:02x}' for b in raw[2:18])}")
        
        # Read the SECOND block (bytes 66-131)
        f.seek(offset + 66)
        raw2 = f.read(66)
        print(f"  Second block (offset 66):")
        print(f"    d (fp16): {raw2[0]:02x} {raw2[1]:02x}")
        
        # Total raw size for this tensor:
        # Each position (k,j) needs 66 bytes for 256 expert values
        # Number of (k,j) positions = 2048 * 512 = 1,048,576
        # Total raw size = 1,048,576 * 66 = 69,206,016 bytes
        total_raw = 2048 * 512 * 66
        print(f"  Expected total raw size: {total_raw} bytes ({total_raw/1024/1024:.1f} MB)")
        
        # Check: does the data at offset 66 correspond to (k=0, j=1) or (k=1, j=0)?
        # Since dims are [2048, 512, 256] and C-order means last dim varies fastest:
        # Block index = (k * 512 * 256/256) + (j * 256/256) + (e/256)
        # = k * 512 + j (since each block covers 256 elements = 1 expert index unit)
        # So block 0 = (k=0, j=0), block 1 = (k=0, j=1), ..., block 512 = (k=1, j=0)
        
        # Let's verify this by checking if block at step 512 makes sense as (k=1, j=0)
        offset_512 = offset + 512 * 66
        f.seek(offset_512)
        raw512 = f.read(66)
        print(f"  Block 512 (offset {512*66} = (k=1,j=0)):")
        print(f"    d (fp16): {raw512[0]:02x} {raw512[1]:02x}")

f.close()
print("\nDone")
