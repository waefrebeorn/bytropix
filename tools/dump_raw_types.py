#!/usr/bin/env python3
"""Dump raw ggml_type for down_exps tensors only."""
import struct, sys
f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
magic = f.read(4)
version = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]
# Quick-and-dirty KV skip: find data start by searching for 'blk.0'
# Actually just skip KV pairs properly
for i in range(kv_count):
    n = struct.unpack('<Q', f.read(8))[0]
    f.read(n)  # key string
    typ = struct.unpack('<i', f.read(4))[0]
    if typ == 8:  # string
        n = struct.unpack('<Q', f.read(8))[0]
        f.read(n)
    elif typ == 9:  # array
        arr_type = struct.unpack('<i', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            if arr_type == 8:
                n = struct.unpack('<Q', f.read(8))[0]
                f.read(n)
            else:
                f.read(4)
    elif typ in (10, 11):
        f.read(8)
    elif typ == 12:
        f.read(8)
    elif typ == 6:
        f.read(4)
    elif typ == 7:
        f.read(1)
    elif typ in (0, 1):
        f.read(1)
    elif typ in (2, 3):
        f.read(2)
    elif typ in (4, 5):
        f.read(4)
    else:
        print(f'Unknown type {typ}', file=sys.stderr)

# Type name map (CORRECT per llama.cpp)
TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',
         8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',
         14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',
         19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',
         24:'Q4_0_4x4',25:'Q4_0_8x8',26:'Q4_0_4x8',29:'IQ1_M'}

for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    if 'down_exps' in name:
        tname = TYPES.get(ggml_type, f'?{ggml_type}?')
        print(f'  {name:45s} type={ggml_type:3d} ({tname})')
f.close()
