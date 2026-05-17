#!/usr/bin/env python3
"""Dump tensor info: name, type, dims, data_offset, raw_size."""
import struct, sys

f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
magic = f.read(4)
version = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]

# Skip KV metadata
for i in range(kv_count):
    n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    typ = struct.unpack('<i', f.read(4))[0]
    if typ == 8:
        n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif typ == 9:
        arr_type = struct.unpack('<i', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            if arr_type == 8:
                n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
            else: f.read(4)
    elif typ in (10, 11, 12): f.read(8)
    elif typ == 6: f.read(4)
    elif typ == 7: f.read(1)
    elif typ in (0, 1): f.read(1)
    elif typ in (2, 3): f.read(2)
    elif typ in (4, 5): f.read(4)

# Tensor info section starts here
tensor_start = f.tell()

TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',
         9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',
         15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',
         20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',29:'IQ1_M'}

# First pass: collect all info
tensors = []
for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims, ggml_type, data_offset))

# Find data blob offset (min data_offset among all tensors)
if tensors:
    data_blob_offset = min(t[4] for t in tensors)
    print(f"Data blob offset: {data_blob_offset}", file=sys.stderr)

# Compute raw_size from ggml_type
def gguf_raw_size(ggml_type, n_elems):
    sizes = {0:4, 1:2, 2:18, 3:20, 6:19, 7:21, 8:34, 9:36,
             10:24, 11:28, 12:144, 13:176, 14:192, 15:208,
             16:36, 17:36, 18:44, 19:24, 20:20, 21:38, 22:36, 23:136, 29:22}
    blck_sizes = {0:1, 1:1, 2:32, 3:32, 6:32, 7:32, 8:32, 9:32,
                  10:256, 11:256, 12:256, 13:256, 14:256, 15:256,
                  16:256, 17:256, 18:256, 19:256, 20:32, 21:256, 22:256, 23:256, 29:256}
    blck_size = blck_sizes.get(ggml_type, 1)
    bsize = sizes.get(ggml_type, 4)
    n_blocks = (n_elems + blck_size - 1) // blck_size
    return n_blocks * bsize

# Filter for search
search = sys.argv[1] if len(sys.argv) > 1 else None

for name, n_dims, dims, ggml_type, data_offset in tensors:
    if search and search not in name:
        continue
    
    n_elems = 1
    for d in dims:
        n_elems *= d
    raw_sz = gguf_raw_size(ggml_type, n_elems)
    file_offset = data_blob_offset + data_offset
    
    shape = '×'.join(str(d) for d in dims)
    tname = TYPES.get(ggml_type, f'?{ggml_type}?')
    print(f"{name:50s} [{shape:20s}] type={ggml_type:3d} {tname:8s} "
          f"file_off={file_offset:12d} raw_sz={raw_sz:12d}")

f.close()
