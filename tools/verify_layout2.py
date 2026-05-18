#!/usr/bin/env python3
"""
Verify the correct tensor layout by examining ALL tensor dimensions and comparing with how
llama.cpp stores them (ne[0] = innermost, read first from GGUF file).

Key insight: In llama.cpp, GGUF stores dimensions with ne[0] first.
ne[0] = innermost (fastest varying) dimension.

For ffn_gate_exps.weight: dims stored in GGUF = [2048, 512, 256]
  - ne[0] = 2048 = D_MODEL
  - ne[1] = 512  = D_FF  
  - ne[2] = 256  = N_EXPERTS
  - Element at (expert e, dff j, dmodel k) = e * 512 * 2048 + j * 2048 + k
  - Expert e's weights ARE contiguous!

Let's verify by checking block offsets.
"""
import struct, numpy as np

f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4)
ver = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]

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

tensors = []
for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims_gguf = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims_gguf, ggml_type, data_offset))

# Find relevant tensors
for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.attn_qkv.weight' in name:
        print(f"attn_qkv.weight: GGUF dims = {dims}")
        print(f"  => ne[0]={dims[0]}, ne[1]={dims[1]}")
        print(f"  => Row-major: element (row={1}, col={0}) = offset {dims[0]} (next row)")
        print(f"  => Our code uses: weight[i + j * {dims[0]}]")
        print(f"     where i=input_idx, j=output_idx")
        print(f"     = {dims[0]} output stride = correct for ne[0]=D_MODEL!")
        
    if 'blk.0.output_norm.weight' in name and gtype == 0:
        print(f"\noutput_norm.weight (F32): dims = {dims}, offset={offset}")
        f.seek(offset)
        vals = struct.unpack(f'<{dims[0]}f', f.read(dims[0]*4))
        print(f"  First 8 values: {vals[:8]}")
        print(f"  Stats: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, min={min(vals):.6f}, max={max(vals):.6f}")
        # This is an RMSNorm weight, should look reasonable
    
    if 'blk.0.ffn_gate_inp.weight' in name and gtype == 0:
        print(f"\nffn_gate_inp.weight (F32): dims = {dims}, offset={offset}")
        print(f"  => ne[0]={dims[0]}=D_MODEL, ne[1]={dims[1]}=N_EXPERTS")
        print(f"  => element at (expert e, dmodel k) = e * {dims[0]} + k")
        
        # Read some data and check
        n_elems = dims[0] * dims[1]
        f.seek(offset)
        data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
        
        # Check basic statistics
        finite = data[np.isfinite(data)]
        print(f"  Total elements: {n_elems}")
        print(f"  Finite elements: {len(finite)} / {n_elems}")
        print(f"  Finite range: [{finite.min():.4f}, {finite.max():.4f}]")
        print(f"  Finite mean: {finite.mean():.4f}, std: {finite.std():.4f}")
        print(f"  NaN count: {np.isnan(data).sum()}")
        print(f"  Inf count: {np.isinf(data).sum()}")
        
        # Check expert 0's weights (contiguous at offset 0..D_MODEL-1)
        expert0 = data[0:dims[0]]
        print(f"  Expert 0 first 8 weights: {expert0[:8].tolist()}")
        print(f"  Expert 0 range: [{expert0.min():.4f}, {expert0.max():.4f}]")
        
        # Check expert 1's weights (contiguous at offset D_MODEL..2*D_MODEL-1)
        expert1 = data[dims[0]:2*dims[0]]
        print(f"  Expert 1 first 8 weights: {expert1[:8].tolist()}")
        print(f"  Expert 1 range: [{expert1.min():.4f}, {expert1.max():.4f}]")

    if 'blk.0.ffn_gate_inp_shexp.weight' in name and gtype == 0:
        print(f"\nffn_gate_inp_shexp.weight (F32): dims = {dims}, offset={offset}")
        f.seek(offset)
        vals = struct.unpack(f'<{dims[0]}f', f.read(dims[0]*4))
        print(f"  First 8 values: {vals[:8]}")
        print(f"  Range: [{min(vals):.4f}, {max(vals):.4f}]")

f.close()
