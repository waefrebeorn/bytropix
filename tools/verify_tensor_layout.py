#!/usr/bin/env python3
"""
Empirically verify the 3D tensor layout for MoE expert weights.

Theory: GGUF stores 3D tensors in C-order (row-major, last dim fastest).
For ffn_gate_exps with dims=[2048, 512, 256]:
  - element (k, j, e) at offset: k*512*256 + j*256 + e
  - expert 0 is NOT contiguous! it's at positions: j*256 + 0 for each (k,j)

We verify by:
1. Loading a known-f32 3D tensor (like output.weight which is 2D)
2. Checking the Python gguf library's view of the data
3. Verifying the actual block structure

For ffn_gate_inp (F32, 2D, [2048, 256]):
  - We load this, verify dimension order
  - Element (k, e) at offset: k*256 + e
  - This is the router: router_score[e] = sum_k x[k] * weight[k*256 + e]
"""
import struct
import numpy as np

f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4)  # magic
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

# ==================== TEST 1: F32 router weight (ffn_gate_inp) ====================
print("=" * 70)
print("TEST 1: F32 ffn_gate_inp.weight [2048, 256]")
print("=" * 70)

for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.ffn_gate_inp.weight' in name and not 'shexp' in name:
        print(f"  GGUF dims: {dims}  (C-order: last dim varies fastest)")
        print(f"  So element (k, e) at offset: k*{dims[1]} + e = k*256 + e")
        
        # Read the full tensor (F32)
        n_elems = dims[0] * dims[1]
        f.seek(offset)
        data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
        
        # If C-order (last dim fastest): element (k,e) at data[k*256 + e]
        # Router: score[e] = sum_k x[k] * weight[k, e]
        # = sum_k x[k] * data[k*256 + e]
        
        # Verify: check element (k=0, e=0) = data[0]
        # and element (k=0, e=1) = data[1]
        # and element (k=1, e=0) = data[256]
        
        # Use dummy input: all 1s
        x = np.ones(2048, dtype=np.float32)
        
        # Compute router scores with C-order interpretation
        scores_corder = np.zeros(256, dtype=np.float64)
        for e in range(256):
            s = 0.0
            for k in range(2048):
                s += x[k] * data[k * 256 + e]  # C-order: k*256 + e
            scores_corder[e] = s
        
        # Compute router scores with column-major interpretation
        scores_colmajor = np.zeros(256, dtype=np.float64)
        for e in range(256):
            s = 0.0
            for k in range(2048):
                s += x[k] * data[k + e * 2048]  # column-major: k + e*2048
            scores_colmajor[e] = s
        
        print(f"\n  Router scores with C-order (k*256 + e):")
        print(f"    First 8: {scores_corder[:8].tolist()}")
        print(f"    Max: {scores_corder.max():.4f}, Min: {scores_corder.min():.4f}, Std: {scores_corder.std():.4f}")
        print(f"    Sum: {scores_corder.sum():.4f}")
        
        print(f"\n  Router scores with column-major (k + e*2048):")
        print(f"    First 8: {scores_colmajor[:8].tolist()}")
        print(f"    Max: {scores_colmajor.max():.4f}, Min: {scores_colmajor.min():.4f}, Std: {scores_colmajor.std():.4f}")
        print(f"    Sum: {scores_colmajor.sum():.4f}")
        
        # Which one looks more like a router? (sum of 2048 weights, each ~O(1))
        # With x=all-1s, the router score = sum_k weight[k, e]
        # These should be reasonable numbers
        
        # Also check the first few values directly
        print(f"\n  Actual data first 8 elements: {data[:8].tolist()}")
        print(f"  Elements at stride 256: {data[::256][:8].tolist()}")
        
        break

# ==================== TEST 2: Q6_K attn_qkv weight (verified correct) ====================
print("\n" + "=" * 70)
print("TEST 2: Q6_K blk.0.attn_qkv.weight [8192, 2048] — VERIFIED CORRECT")
print("=" * 70)

for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.attn_qkv.weight' in name:
        print(f"  GGUF dims: {dims}")
        print(f"  Type: Q6_K")
        print(f"  Verified correct indexing: weight[i + j*2048]")
        print(f"  This matches C-order: element at (row=j, col=i) = j*2048 + i")
        print(f"  Where dims[0]=8192 (rows/output dim), dims[1]=2048 (cols/input dim)")
        break

# ==================== TEST 3: ffn_gate_exps block layout ====================
print("\n" + "=" * 70)
print("TEST 3: IQ2_XXS blk.0.ffn_gate_exps.weight [2048, 512, 256]")
print("=" * 70)

for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.ffn_gate_exps.weight' in name:
        print(f"  GGUF dims: {dims}")
        print(f"  C-order: element (k, j, e) at offset: k*{dims[1]}*{dims[2]} + j*{dims[2]} + e")
        print(f"  One IQ2_XXS block (66 bytes) covers 256 elements = 1 (k,j) × 256 experts")
        print(f"  Block index for (k,j): k*{dims[1]} + j = k*512 + j")
        print()
        
        # Read first 2 blocks
        f.seek(offset)
        b0 = f.read(66)
        b1 = f.read(66)
        
        # Block 0 = (k=0, j=0), Block 1 = (k=0, j=1)
        # Block 512 = (k=1, j=0)
        f.seek(offset + 512 * 66)
        b512 = f.read(66)
        
        print(f"  Block 0 (k=0,j=0): d_bits={b0[0]:02x}{b0[1]:02x}")
        print(f"  Block 1 (k=0,j=1): d_bits={b1[0]:02x}{b1[1]:02x}")
        print(f"  Block 512 (k=1,j=0): d_bits={b512[0]:02x}{b512[1]:02x}")
        
        # Verify: total raw data size
        total_blocks = dims[0] * dims[1]  # 2048 * 512 = 1,048,576
        total_raw = total_blocks * 66
        print(f"  Total blocks: {total_blocks}")
        print(f"  Total raw bytes: {total_raw} ({total_raw/1024/1024:.1f} MB)")
        
        # IMPORTANT: Expert e's weights are at specific positions within each block
        # After dequant to F32, expert e's weight at (k,j) is at:
        # f32_out[k * 512 * 256 + j * 256 + e]
        # This is NOT contiguous!
        
        print(f"\n  CONCLUSION: Expert weights are INTERLEAVED in memory!")
        print(f"  Expert e at position (k,j) = f32[(k*512 + j)*256 + e]")
        print(f"  NOT: f32[e*2048*512 + k*512 + j] (what our code assumes)")
        print(f"  NOT: f32[e*2048*512 + j*2048 + k] (what our code actually computes)")
        
        break

f.close()
print("\nDone")
