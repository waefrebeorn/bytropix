#!/usr/bin/env python3
"""Dump ggml_types for all MoE tensors across all layers."""
import struct
f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4)  # magic
struct.unpack('<I', f.read(4))  # version
tensor_count = struct.unpack('<q', f.read(8))[0]
kv_count = struct.unpack('<q', f.read(8))[0]
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

moe_tensors = {}  # name -> {types: set, type: int}

for i in range(tensor_count):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    
    # Extract base name (remove blk.X prefix)
    base = name.split('.', 2)[-1] if 'blk.' in name else name
    
    if any(x in base for x in ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 
                                'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp',
                                'ffn_gate_inp', 'ffn_gate_inp_shexp']):
        tname = TYPES.get(ggml_type, f'?{ggml_type}?')
        shape = '×'.join(str(d) for d in dims)
        
        if base not in moe_tensors:
            moe_tensors[base] = {'types': set(), 'shapes': set(), 'type_counts': {}}
        moe_tensors[base]['types'].add(tname)
        moe_tensors[base]['shapes'].add(shape)
        key = f"type={ggml_type} {tname}"
        moe_tensors[base]['type_counts'][key] = moe_tensors[base]['type_counts'].get(key, 0) + 1

print("=== MoE Tensor Type Summary ===")
for base, info in sorted(moe_tensors.items()):
    print(f"\n{base}:")
    print(f"  Shapes: {', '.join(sorted(info['shapes']))}")
    print(f"  Types: {', '.join(sorted(info['types']))}")
    for tc, cnt in sorted(info['type_counts'].items()):
        print(f"    {tc}: {cnt} tensors")
f.close()
