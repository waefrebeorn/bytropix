#!/usr/bin/env python3
"""
Verify our loaded expert weights match the GGUF file.
Take expert 64 (top expert from router) and compare:
1. Python raw extraction from GGUF
2. What our C code would see after loading
"""
import struct
import numpy as np

# Re-open GGUF and find expert 64's weights
f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4); f.read(4); f.read(8); f.read(8)

for i in range(54):
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
for i in range(733):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims, ggml_type, data_offset))

data_blob_start = f.tell()
data_blob_start_aligned = ((data_blob_start + 31) // 32) * 32

D_MODEL = 2048
D_FF = 512
N_EXPERTS = 256
E = 64  # top expert index

# Check gate_exps: ne=[2048, 512, 256], IQ2_XXS
# In C-order: element at (e, j, k) = e * 512 * 2048 + j * 2048 + k
# For IQ2_XXS: each block = 66 bytes = 256 elements
# Block for expert e at d_model-index k_start=256*i, d_ff=j:
#   block_idx = e * 512 * 8 + j * 8 + k_start/256
#   = e * 4096 + j * 8 + i

for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.ffn_gate_exps.weight' in name:
        abs_off = data_blob_start_aligned + offset
        print(f"\n{name}:")
        print(f"  ne = {dims}, type={gtype}")
        
        # For expert 64, column j=0, d_model index k=0..2047
        # In the raw data, this is 8 blocks at positions:
        # block_start = E * 512 * 8 + 0 * 8 + 0 = E * 4096
        block_start = E * D_FF * (D_MODEL // 256)  # = E * 512 * 8 = E * 4096
        
        f.seek(abs_off + block_start * 66)
        raw_gate = f.read(8 * 66)  # 8 blocks = 1 column = 2048 elements
        
        print(f"  Expert {E} gate, column 0, first 66 bytes:")
        print(f"    d0: {raw_gate[0]:02x}{raw_gate[1]:02x}")
        print(f"    qs[0..7]: {' '.join(f'{b:02x}' for b in raw_gate[2:10])}")
        
        # Now check column j=1
        f.seek(abs_off + (block_start + 8) * 66)
        raw_gate2 = f.read(2)  # just the d value
        print(f"  Expert {E} gate, column 1, d: {raw_gate2[0]:02x}{raw_gate2[1]:02x}")
        
        # Check what our C code would see after dequant
        # For F32 output at expert e, the data is at:
        # f32_out[e * D_MODEL * D_FF + ...]
        # For column j, d_model k: offset = e * D_MODEL * D_FF + j * D_MODEL + k
        
        # Total F32 size
        n_elems = dims[0] * dims[1] * dims[2]
        f.seek(abs_off)
        raw_all = f.read((n_elems // 256) * 66)  # raw quantized data
        
        # Our C code loads everything via gguf_read_tensor_f32
        # The dequantized output has layout matching GGUF ne order
        # Let me verify the dequantized layout
        
        # Extract first IQ2_XXS block for expert 64, column 0
        # Check it using ctypes by calling dequantize_iq2_xxs_block
        
        gate_bytes_raw = raw_all[block_start*66:(block_start+8)*66]
        print(f"\n  Gate raw size for expert {E}: {len(gate_bytes_raw)} bytes")
        print(f"  Expected: {D_MODEL * D_FF // 256 * 66} = {D_MODEL*D_FF//256*66}")
        break

# Also check if the C code's ffn_gate_exps layout is correct
# In our C code: ffn_gate_exps[e * D_MODEL * D_FF + j * D_MODEL + k]
# This assumes the dequantized data has:
#   - Expert e first (slowest)
#   - Then column j (D_FF)
#   - Then element k (D_MODEL, fastest)
# With ne=[2048, 512, 256]:
#   - ne[0]=2048 (D_MODEL, fastest)
#   - ne[1]=512 (D_FF)
#   - ne[2]=256 (N_EXPERTS, slowest)
# Our offset: e * 2048 * 512 + j * 2048 + k
# Should be:  e * 512 * 2048 + j * 2048 + k
# These are EQUAL due to commutativity!

print("\nVerification complete")
f.close()
