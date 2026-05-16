#!/usr/bin/env python3
"""Dump raw ggml_type for gate/up_exps and common tensors."""
import struct
f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4)  # magic
struct.unpack('<I', f.read(4))  # version
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]
# Skip KV pairs (as before)
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

TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',
         9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',
         15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',
         20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',29:'IQ1_M'}

targets = ['gate_exps', 'up_exps', 'gate_inp', 'attn_qkv', 'attn_q', 'attn_k', 'attn_v', 
           'attn_output', 'attn_gate', 'ssm_out', 'output.weight', 'token_embd']

for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    # Only show first layer + any interesting types
    layer_match = False
    for t in targets:
        if t in name and ('blk.0.' in name or 'output' in name or 'token_embd' in name):
            layer_match = True
    if layer_match or (i < 3):
        tname = TYPES.get(ggml_type, f'?{ggml_type}?')
        shape = '×'.join(str(d) for d in dims)
        print(f'  {name:45s} [{shape:20s}] type={ggml_type:3d} {tname}')
f.close()
