import struct

def read_string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

f = open('/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
assert f.read(4) == b'GGUF'
version = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]
n_kv = struct.unpack('<Q', f.read(8))[0]

print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV")

for i in range(n_kv):
    key = read_string(f)
    typ = struct.unpack('<I', f.read(4))[0]
    if typ == 8:
        val = read_string(f)
    elif typ == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        vals = []
        for _ in range(arr_len):
            if arr_type == 8: vals.append(read_string(f))
            elif arr_type == 4: vals.append(struct.unpack('<I', f.read(4))[0])
            elif arr_type == 10: vals.append(struct.unpack('<Q', f.read(8))[0])
            else: vals.append(None)
        val = vals
    elif typ == 4: val = struct.unpack('<I', f.read(4))[0]
    elif typ == 10: val = struct.unpack('<Q', f.read(8))[0]
    elif typ == 0: val = struct.unpack('<B', f.read(1))[0]
    else: f.read(1); continue
    
    keywords = ['expert', 'moe', 'ffn', 'hidden', 'intermediate', 'n_layer', 'dense', 'n_embd', 'n_head']
    if any(k in key.lower() for k in keywords):
        print(f'  {key}: {val}')

pad = (f.tell() + 31) & ~31
f.seek(pad)

print('\nFFN/Expert tensors:')
ffn_count = 0
for i in range(n_tensors):
    n_dims = struct.unpack('<I', f.read(4))[0]
    name = read_string(f)
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<I', f.read(4))[0]
    offset = struct.unpack('<Q', f.read(8))[0]
    if 'ffn' in name.lower() or 'expert' in name.lower():
        print(f'  {name}: dims={dims} type={ggml_type}')
        ffn_count += 1
f.close()
print(f'Total: {ffn_count}')
