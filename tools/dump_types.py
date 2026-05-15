#!/usr/bin/env python3
"""Quick GGUF tensor type dump - handles all GGUF v3 KV types"""
import struct, sys

f = open('/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
magic = f.read(4)
if magic != b'GGUF':
    print(f"Bad magic: {magic}")
    sys.exit(1)

ver = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]
n_kv = struct.unpack('<Q', f.read(8))[0]
print(f"GGUF v{ver}, {n_tensors} tensors, {n_kv} KV pairs")

TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
         10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
         16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',21:'IQ3_S',22:'IQ2_S',23:'IQ1_M'}

# Skip KV pairs properly
for i in range(n_kv):
    klen = struct.unpack('<I', f.read(4))[0]
    f.read(klen)
    vtype = struct.unpack('<I', f.read(4))[0]
    if vtype == 0:  # uint32
        f.read(4)
    elif vtype == 1:  # int32
        f.read(4)
    elif vtype == 2:  # float32
        f.read(4)
    elif vtype == 3:  # bool
        f.read(1)
    elif vtype == 4:  # string
        sl = struct.unpack('<Q', f.read(8))[0]
        f.read(sl)
    elif vtype == 5:  # array
        arrtype = struct.unpack('<I', f.read(4))[0]
        arrlen = struct.unpack('<Q', f.read(8))[0]
        for a in range(arrlen):
            if arrtype == 0: f.read(4)
            elif arrtype == 1: f.read(4)
            elif arrtype == 2: f.read(4)
            elif arrtype == 3: f.read(1)
            elif arrtype == 4: 
                sl2 = struct.unpack('<Q', f.read(8))[0]
                f.read(sl2)
            elif arrtype in (6,7): f.read(8)
            elif arrtype == 8: f.read(8)
    elif vtype == 6:  # uint64
        f.read(8)
    elif vtype == 7:  # int64
        f.read(8)
    elif vtype == 8:  # float64
        f.read(8)

# Read tensor info
targets = ['output.weight', 'token_embd.weight', 'blk.0.attn_qkv.weight', 
           'blk.0.attn_gate.weight', 'blk.0.ssm_out.weight',
           'blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight',
           'blk.0.ffn_gate_inp.weight']

for i in range(n_tensors):
    nlen = struct.unpack('<I', f.read(4))[0]
    name = f.read(nlen).decode('utf-8')
    n_dim = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for d in range(n_dim)]
    gt = struct.unpack('<I', f.read(4))[0]
    data_off = struct.unpack('<Q', f.read(8))[0]
    if name in targets:
        tname = TYPES.get(gt, f'UK({gt})')
        print(f'{name}: type={gt}({tname}) dims={dims}')

f.close()
