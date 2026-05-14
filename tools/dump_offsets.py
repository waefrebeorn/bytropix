#!/usr/bin/env python3
"""Dump tensor order and offsets from GGUF file, focusing on block alignment."""
import struct
import sys

path = sys.argv[1]
with open(path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    # Skip KV metadata
    for _ in range(n_kv):
        klen = struct.unpack('<Q', f.read(8))[0]
        f.read(klen)
        vtype = struct.unpack('<I', f.read(4))[0]
        if vtype == 8:
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            for _ in range(arr_len):
                if arr_type == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                else:
                    f.read(4)
        elif vtype in (4, 5, 6):
            f.read(4)
        elif vtype in (0, 1, 7):
            f.read(1)
        elif vtype in (2, 3):
            f.read(2)
        else:
            f.read(4)
    
    tensor_end = f.tell()
    alignment = 128
    pad = (alignment - (tensor_end % alignment)) % alignment
    data_blob_offset = tensor_end + pad
    
    print("Tensor info table end:", tensor_end)
    print("Data blob offset:", data_blob_offset)
    print(f"Alignment padding: {pad}")
    
    TYPE_NAMES = {0: "F32", 1: "F16", 10: "Q4_0", 14: "Q5_K", 15: "Q6_K", 
                  16: "Q2_K", 17: "Q3_K", 18: "IQ2_S", 19: "IQ2_XXS", 
                  20: "IQ1_S", 21: "IQ2_XS", 26: "Q4_K", 30: "IQ1_M"}
    
    tensors = []
    for i in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack('<I', f.read(4))[0]
        data_offset = struct.unpack('<Q', f.read(8))[0]
        
        n_elems = 1
        for d in dims:
            n_elems *= d
        
        type_name = TYPE_NAMES.get(ggml_type, f"T{ggml_type}")
        tensors.append((data_offset, name, type_name, ggml_type, dims, n_elems))
    
    # Sort by data_offset
    tensors.sort(key=lambda t: t[0])
    
    print(f"\nAll tensors ({len(tensors)} total):")
    print(f"{'idx':>4} {'offset':>12} {'size':>12} {'type':>8} {'name':<50} {'dims'}")
    print("-" * 120)
    
    for i, (off, name, tname, tid, dims, nelems) in enumerate(tensors):
        # Compute raw size based on type
        bk_size_map = {
            0: 4,      # F32
            1: 2,      # F16
            10: 36,    # Q4_0 = d(fp16,2) + qs[32] = 34 bytes per 32 elems... 
            14: 80,    # Q5_K: 256 elems per block (approx)
            15: 210,   # Q6_K
            16: 64,    # Q2_K
            17: 104,   # Q3_K
            18: 82,    # IQ2_S
            19: 66,    # IQ2_XXS
            20: 50,    # IQ1_S
            21: 74,    # IQ2_XS
            26: 176,   # Q4_K
            30: 44,    # IQ1_M
        }
        bpk = bk_size_map.get(tid, 0)  # bytes per block (of 256 elems for K-quant)
        if bpk > 0:
            n_blocks = (nelems + 255) // 256
            raw_size = n_blocks * bpk
        else:
            raw_size = nelems * 4  # F32 fallback
            n_blocks = nelems
        
        if i < 5 or i > len(tensors) - 5 or tid == 18 or (i > 0 and tensors[i-1][3] == 18) or (i > 0 and tensors[i+1][3] == 18):
            print(f"{i:>4} {off:>12} {raw_size:>12} {tname:>8} {name:<50} {dims}")
    
    # Show tensors around IQ2_S boundaries
    print("\n\n=== Tensors around IQ2_S boundaries ===")
    prev_data_end = 0
    for i, (off, name, tname, tid, dims, nelems) in enumerate(tensors):
        bpk = bk_size_map.get(tid, 0)
        if bpk > 0:
            n_blocks = (nelems + 255) // 256
            raw_size = n_blocks * bpk
        else:
            raw_size = nelems * 4
        data_end = off + raw_size
        
        gap = off - prev_data_end if i > 0 else 0
        if gap != 0:
            print(f"  GAP at tensor {i}: prev_end={prev_data_end}, this_off={off}, gap={gap}")
        
        if tid == 18 or (i > 0 and tensors[i-1][3] == 18):
            print(f"{i:>4} off={off:>10} end={data_end:>10} sz={raw_size:>10} {tname:>8} {name:<50}")
        
        prev_data_end = data_end
