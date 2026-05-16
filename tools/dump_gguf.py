#!/usr/bin/env python3
"""Standalone GGUF reader: dump tensor names, shapes, and metadata from a GGUF file.
No dependencies beyond Python stdlib."""

import struct
import sys
import os

# GGML type names — verified against gguf_reader.h
GGML_TYPE = {
    0: ("F32", 4), 1: ("F16", 2), 2: ("Q4_0", 0), 3: ("Q4_1", 0),
    6: ("Q5_0", 0), 7: ("Q5_1", 0), 8: ("Q8_0", 0), 9: ("Q8_1", 0),
    10: ("Q2_K", 0), 11: ("Q3_K", 0), 12: ("Q4_K", 0), 13: ("Q5_K", 0),
    14: ("Q6_K", 0), 15: ("Q8_K", 0), 16: ("IQ2_XXS", 0), 17: ("IQ2_XS", 0),
    18: ("IQ3_XXS", 0), 19: ("IQ1_S", 0), 20: ("IQ4_NL", 0),
    21: ("IQ3_S", 0), 22: ("IQ2_S", 0), 23: ("IQ4_XS", 0),
    24: ("Q4_0_4x4", 0), 25: ("Q4_0_8x8", 0), 26: ("Q4_0_4x8", 0),
    29: ("IQ1_M", 0),
}

# GGUF types
GGUF_TYPE_NAMES = ['UINT8', 'INT8', 'UINT16', 'INT16', 'UINT32', 'INT32', 'FLOAT32', 'BOOL', 'STRING', 'ARRAY', 'UINT64', 'INT64', 'FLOAT64']

def read_string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def read_value(f, typ):
    if typ == 0: return struct.unpack('<B', f.read(1))[0]
    elif typ == 1: return struct.unpack('<b', f.read(1))[0]
    elif typ == 2: return struct.unpack('<H', f.read(2))[0]
    elif typ == 3: return struct.unpack('<h', f.read(2))[0]
    elif typ == 4: return struct.unpack('<I', f.read(4))[0]
    elif typ == 5: return struct.unpack('<i', f.read(4))[0]
    elif typ == 6: return struct.unpack('<f', f.read(4))[0]
    elif typ == 7: return struct.unpack('<?', f.read(1))[0]
    elif typ == 8: return read_string(f)
    elif typ == 9:  # array
        arr_type = struct.unpack('<i', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif typ == 10: return struct.unpack('<Q', f.read(8))[0]
    elif typ == 11: return struct.unpack('<q', f.read(8))[0]
    elif typ == 12: return struct.unpack('<d', f.read(8))[0]
    return None

def dump_gguf(path):
    f = open(path, 'rb')
    magic = f.read(4)
    if magic != b'GGUF':
        print("Not a GGUF file")
        return
    
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<q', f.read(8))[0]
    kv_count = struct.unpack('<q', f.read(8))[0]
    
    print(f"GGUF version: {version}")
    print(f"Tensor count: {tensor_count}")
    print(f"KV pairs: {kv_count}")
    print()
    
    # Read KV pairs
    alignment = 32  # default
    for i in range(kv_count):
        key = read_string(f)
        typ = struct.unpack('<i', f.read(4))[0]
        val = read_value(f, typ)
        typ_name = GGUF_TYPE_NAMES[typ] if typ < len(GGUF_TYPE_NAMES) else f"TYPE_{typ}"
        
        if isinstance(val, str):
            print(f"  KV[{i}] {key} ({typ_name}) = \"{val[:80]}{'...' if len(val)>80 else ''}\"")
        elif isinstance(val, list):
            print(f"  KV[{i}] {key} ({typ_name}) = [{', '.join(str(v)[:20] for v in val[:5])}{'...' if len(val)>5 else ''}]")
        else:
            print(f"  KV[{i}] {key} ({typ_name}) = {val}")
        
        if key == 'general.alignment':
            alignment = val
    
    print()
    print(f"Alignment: {alignment}")
    print()
    
    # Find data offset (where tensor data starts)
    # After KV pairs, we have tensor info, then the data blob
    data_start = f.tell()
    
    # Read tensor info
    tensors = []
    for i in range(tensor_count):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack('<i', f.read(4))[0]
        data_offset = struct.unpack('<Q', f.read(8))[0]
        
        type_name = GGML_TYPE.get(ggml_type, (f"TYPE_{ggml_type}", 0))[0]
        tensors.append((name, dims, ggml_type, type_name, data_offset))
        
        shape_str = '×'.join(str(d) for d in dims)
        print(f"  TENSOR[{i:3d}] {name:45s} [{shape_str:20s}] {type_name:8s} offset={data_offset:12d}")

    # Data blob start
    # After all tensor info, aligned to alignment boundary
    data_actual_start = f.tell()
    # The tensor data offsets are relative to some base, find it
    # Usually the first tensor has offset 0 or offset relative to data start
    if tensors:
        first_off = tensors[0][4]
        print(f"\n  First tensor offset: {first_off}")
        print(f"  Data section starts at: {data_actual_start}")
        # The data_offset is from the START of the data blob
        data_blob_start = data_actual_start
        if first_off > 0:
            # Offsets might be relative to end of header
            data_blob_start = f.tell()  # already at end of tensor info
            # Pad to alignment
            pad = (alignment - (data_blob_start % alignment)) % alignment
            data_blob_start += pad
        
        # Calculate where first tensor actually lives
        # In GGUF v3, tensor data offset is relative to start of the data blob
        first_data_pos = data_blob_start + first_off
        
        print(f"\n  Data blob starts at file pos: {data_blob_start}")
        print(f"  First tensor data at: {first_data_pos}")
        
        # Check file size
        f.seek(0, 2)
        file_size = f.tell()
        print(f"  File size: {file_size} ({file_size/1e9:.1f}GB)")
        
        # Calculate size of first tensor
        name = tensors[0][0]
        dims = tensors[0][1]
        ggml_type_id = tensors[0][2]
        type_name = tensors[0][3]
        
        # For quantized types, estimate element size
        if 'IQ2' in type_name:
            # IQ2_M: 2-bit, 4-bit scale per block of 32
            elem_bytes = 0.25 + (4/32)  # ~0.375 bytes per element
        elif 'Q4' in type_name:
            elem_bytes = 0.5 + (4/32)   # ~0.625 bytes per element  
        else:
            elem_bytes = GGML_TYPE.get(ggml_type_id, (None, 4))[1]
        
        n_elems = 1
        for d in dims:
            n_elems *= d
        
        if elem_bytes:
            estimated_bytes = n_elems * elem_bytes
            print(f"  {name}: {n_elems} elements × ~{elem_bytes:.4f}B = ~{estimated_bytes/1e6:.1f}MB")
    
    f.close()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    dump_gguf(path)
