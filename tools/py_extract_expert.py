#!/usr/bin/env python3
"""
Compute reference expert output for expert 64 using Python (from GGUF raw data).
Compare against our C code's output.
"""
import struct
import numpy as np

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

# Load MoE input
moe_input = np.frombuffer(open('/tmp/dbg_moe_input.bin','rb').read(), dtype=np.float32)
print(f"MoE input: mean={moe_input.mean():.6f} std={moe_input.std():.6f}")

D_MODEL = 2048
D_FF = 512
N_EXPERTS = 256

# Expert 64
E = 64

# ============================================================
# 1. Load gate_exps weights for expert 64
# ============================================================
for name, nd, dims, gtype, offset in tensors:
    if 'blk.0.ffn_gate_exps.weight' in name:
        abs_off = data_blob_start_aligned + offset
        n_elems = dims[0] * dims[1] * dims[2]  # 2048 * 512 * 256
        total_blocks = n_elems // 256
        raw_all = np.frombuffer(open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb').read(), dtype=np.uint8)
        
        # Extract raw blocks for expert E
        # Block layout: ne=[2048, 512, 256]
        # For (e, j, i) where e=expert, j=d_ff, i=d_model:
        #   block_idx = e * 512 * 8 + j * 8 + i//256
        #   byte_offset = block_idx * 66
        
        def get_block(name, dims, e, j, i_block):
            """Get byte offset for IQ2_XXS block at (expert e, d_ff j, d_model block i_block)"""
            # dims=[2048, 512, 256], ne[0]=2048, ne[1]=512, ne[2]=256
            # block_idx = e * dims[1] * (dims[0]//256) + j * (dims[0]//256) + i_block
            blocks_per_col = dims[0] // 256
            block_idx = e * dims[1] * blocks_per_col + j * blocks_per_col + i_block
            return block_idx
        
        break

print("Using raw IQ2_XXS block extraction")
print(f"Total blocks: {total_blocks}")
print(f"Blocks per expert: {D_MODEL * D_FF // 256}")

# We need to use gguf library to dequantize
# Let me try a different approach: use the Python gguf library which already has dequant

import gguf
reader = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
tensors_by_name = {t.name: t for t in reader.tensors}

# Get gate_exps for blk.0
gate_exps = tensors_by_name['blk.0.ffn_gate_exps.weight']
up_exps = tensors_by_name['blk.0.ffn_up_exps.weight']
down_exps = tensors_by_name['blk.0.ffn_down_exps.weight']

print(f"\ngate_exps shape: {gate_exps.shape}, dtype: {gate_exps.data.dtype}")
print(f"gate_exps data shape: {gate_exps.data.shape}")

# The data shape in Python's view: (256, 512, 528)
# Where 256=N_EXPERTS, 512=D_FF, 528 = bytes_per_col
# bytes_per_col = (2048/256) * 66 = 8 * 66 = 528

# So gate_exps.data[E] gives all columns for expert E
# gate_exps.data[E, j] gives column j for expert E (528 bytes)
# Within column: 8 blocks of 66 bytes each

E = 64
gate_raw = gate_exps.data[E].flatten()  # 512 * 528 = 270,336 bytes
up_raw = up_exps.data[E].flatten()
down_raw = down_exps.data[E].flatten()

print(f"Expert {E} gate raw: {len(gate_raw)} bytes")
print(f"  First 16 bytes: {gate_raw[:16].tolist()}")

# But we can't dequantize directly from Python without a dequantizer
# Let me use a different approach: write a small C program that dumps
# expert 64's dequantized weights

# Actually, let me use the gguf library's internal dequant
# The data is raw uint8 - we need to dequantize it

# Let me write the raw data to files and use C to dequantize
open('/tmp/dbg_expert_gate_raw.bin', 'wb').write(bytes(gate_raw))
open('/tmp/dbg_expert_up_raw.bin', 'wb').write(bytes(up_raw))
open('/tmp/dbg_expert_down_raw.bin', 'wb').write(bytes(down_raw))
print(f"\nSaved raw expert {E} data to /tmp/dbg_expert_*_raw.bin")

# For the shared expert
gate_shexp = tensors_by_name['blk.0.ffn_gate_shexp.weight']
up_shexp = tensors_by_name['blk.0.ffn_up_shexp.weight']
down_shexp = tensors_by_name['blk.0.ffn_down_shexp.weight']

print(f"\nShared expert gate type: {gate_shexp.tensor_type}, shape: {gate_shexp.data.shape}")
print(f"Shared expert up type: {up_shexp.tensor_type}, shape: {up_shexp.data.shape}")
print(f"Shared expert down type: {down_shexp.tensor_type}, shape: {down_shexp.data.shape}")

# For Q5_K: data shape is (512, 176) where 176 = d_model_blocks * block_size
# Actually Q5_K block is 176 bytes per 256 elements
# D_MODEL=2048, so 2048/256 = 8 blocks per column
# data[:8*176] = column 0, bytes [8*176:16*176] = column 1, etc.

# Let me just dump raw data and process in C
open('/tmp/dbg_shexp_gate_raw.bin', 'wb').write(bytes(gate_shexp.data.flatten()))
open('/tmp/dbg_shexp_up_raw.bin', 'wb').write(bytes(up_shexp.data.flatten()))
open('/tmp/dbg_shexp_down_raw.bin', 'wb').write(bytes(down_shexp.data.flatten()))

print(f"\nSaved shared expert raw data")
print(f"  gate_shexp raw: {len(gate_shexp.data.flatten())} bytes")
print(f"  up_shexp raw: {len(up_shexp.data.flatten())} bytes")
print(f"  down_shexp raw: {len(down_shexp.data.flatten())} bytes")

f.close()
