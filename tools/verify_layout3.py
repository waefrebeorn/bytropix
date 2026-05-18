#!/usr/bin/env python3
"""
More robust GGUF parsing to compute absolute data offsets.
"""
import struct
import numpy as np

f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')

magic = f.read(4)
version = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<q', f.read(8))[0]
metadata_kv_count = struct.unpack('<q', f.read(8))[0]

print(f"GGUF v{version}, {tensor_count} tensors, {metadata_kv_count} KV pairs")

# Skip KV pairs using the proven method from dump_layer0_types.py
for i in range(metadata_kv_count):
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

tensor_info_start = f.tell()
print(f"Tensor info starts at byte: {tensor_info_start}")

# Read all tensor info
tensors = []
for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims_gguf = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims_gguf, ggml_type, data_offset))

data_blob_start = f.tell()
# Apply GGUF alignment (default = 32 bytes)
alignment = 32
data_blob_start_aligned = ((data_blob_start + alignment - 1) // alignment) * alignment
print(f"Data blob starts (before align): {data_blob_start}, after {alignment}-byte align: {data_blob_start_aligned}")
print(f"  Difference: {data_blob_start_aligned - data_blob_start}")

# Now check the right tensors
for name, nd, dims, gtype, offset in tensors:
    # F32 tensors
    if 'blk.0.ffn_gate_inp.weight' in name and 'shexp' not in name and gtype == 0:
        abs_offset = data_blob_start_aligned + offset
        print(f"\n=== ffn_gate_inp.weight (F32) ===")
        print(f"  ne = {dims}")
        print(f"  blob_offset={offset}, abs_offset={abs_offset}")
        
        n_elems = dims[0] * dims[1]
        f.seek(abs_offset)
        data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
        
        finite = data[np.isfinite(data)]
        print(f"  Total: {n_elems}, Finite: {len(finite)}/{n_elems}")
        print(f"  Finite: min={finite.min():.4f}, max={finite.max():.4f}")
        print(f"  NaN: {np.isnan(data).sum()}, Inf: {np.isinf(data).sum()}")
        
        # According to our analysis: ne[0]=D_MODEL=2048 (innermost)
        # Expert e's weights: data[e * 2048 : (e+1) * 2048]
        e0 = data[0:2048]
        e1 = data[2048:4096]
        print(f"  Expert 0: [{e0[0]:.6f}, {e0[1]:.6f}, ... {e0[7]:.6f}]")
        print(f"  Expert 0 stats: mean={e0.mean():.4f}, std={e0.std():.4f}, range=[{e0.min():.4f}, {e0.max():.4f}]")
        print(f"  Expert 1: [{e1[0]:.6f}, {e1[1]:.6f}, ... {e1[7]:.6f}]")
        print(f"  Expert 1 stats: mean={e1.mean():.4f}, std={e1.std():.4f}, range=[{e1.min():.4f}, {e1.max():.4f}]")
    
    # Q5_K tensors
    if 'blk.0.attn_gate.weight' in name and gtype == 13:
        abs_offset = data_blob_start_aligned + offset
        print(f"\n=== blk.0.attn_gate.weight (Q5_K) ===")
        print(f"  ne = {dims}, blob_offset={offset}, abs_offset={abs_offset}")
        f.seek(abs_offset)
        raw = f.read(32)
        print(f"  First 32 bytes: {' '.join(f'{b:02x}' for b in raw)}")

    # output_norm.weight (F32, 1D)
    if 'output_norm.weight' in name:
        abs_offset = data_blob_start_aligned + offset
        print(f"\n=== output_norm.weight (F32) ===")
        print(f"  ne = {dims}, blob_offset={offset}, abs_offset={abs_offset}")
        f.seek(abs_offset)
        data = np.frombuffer(f.read(dims[0]*4), dtype=np.float32)
        print(f"  First 8: {data[:8].tolist()}")
        print(f"  Range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")

f.close()
