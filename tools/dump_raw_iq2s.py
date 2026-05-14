#!/usr/bin/env python3
"""Dump raw bytes of first IQ2_S tensor from GGUF file and manually dequantize first block."""
import struct
import sys
import os

def read_gguf_tensor_info(f):
    """Read one tensor info entry from GGUF."""
    # Name length + name
    name_len = struct.unpack('<Q', f.read(8))[0]
    name = f.read(name_len).decode('utf-8')
    # n_dims
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = []
    for _ in range(n_dims):
        dims.append(struct.unpack('<Q', f.read(8))[0])
    # ggml_type
    ggml_type = struct.unpack('<I', f.read(4))[0]
    # data_offset
    data_offset = struct.unpack('<Q', f.read(8))[0]
    return name, dims, ggml_type, data_offset

def main():
    path = sys.argv[1]
    with open(path, 'rb') as f:
        # Magic
        magic = f.read(4)
        print(f"Magic: {magic.hex()}")
        
        # Version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")
        
        # Tensor count
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensor count: {n_tensors}")
        
        # Metadata key-value pairs
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"KV pairs: {n_kv}")
        
        # Skip metadata
        for _ in range(n_kv):
            # Key
            klen = struct.unpack('<Q', f.read(8))[0]
            f.read(klen)
            # Value type
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 0:  # uint8
                f.read(1)
            elif vtype == 1:  # int8
                f.read(1)
            elif vtype == 2:  # uint16
                f.read(2)
            elif vtype == 3:  # int16
                f.read(2)
            elif vtype == 4:  # uint32
                f.read(4)
            elif vtype == 5:  # int32
                f.read(4)
            elif vtype == 6:  # float32
                f.read(4)
            elif vtype == 7:  # bool
                f.read(1)
            elif vtype == 8:  # string
                slen = struct.unpack('<Q', f.read(8))[0]
                f.read(slen)
            elif vtype == 9:  # array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    if arr_type == 0: f.read(1)
                    elif arr_type == 1: f.read(1)
                    elif arr_type == 2: f.read(2)
                    elif arr_type == 3: f.read(2)
                    elif arr_type == 4: f.read(4)
                    elif arr_type == 5: f.read(4)
                    elif arr_type == 6: f.read(4)
                    elif arr_type == 7: f.read(1)
                    elif arr_type == 8:
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.read(slen)
            else:
                print(f"Unknown type {vtype}")
                sys.exit(1)
        
        # Read tensor info
        print(f"\n=== Tensor Information ===")
        tensors = []
        for i in range(n_tensors):
            name, dims, ggml_type, data_offset = read_gguf_tensor_info(f)
            tensors.append((name, dims, ggml_type, data_offset))
            if ggml_type == 18:  # IQ2_S
                print(f"  IQ2_S: {name} dims={dims} offset={data_offset}")
        
        # Find data blob start (after all tensor info + alignment padding)
        data_start = f.tell()
        print(f"\nTensors section ends at: {data_start}")
        
        # Find alignment
        # GGUF alignment is typically stored in metadata as "general.alignment"
        alignment = 128  # default llama.cpp alignment
        print(f"Using alignment: {alignment}")
        
        # Compute data blob offset
        pad = (alignment - (data_start % alignment)) % alignment
        data_blob_offset = data_start + pad
        print(f"Data blob offset: {data_blob_offset}")
        
        # Find first IQ2_S tensor
        iq2s_tensors = [(n,d,t,o) for (n,d,t,o) in tensors if t == 18]
        if not iq2s_tensors:
            print("No IQ2_S tensors found!")
            return
        
        name, dims, ggml_type, data_offset = iq2s_tensors[0]
        print(f"\n=== First IQ2_S tensor: {name} ===")
        print(f"  dims={dims}, data_offset={data_offset}")
        print(f"  File position: {data_blob_offset + data_offset}")
        
        n_elems = 1
        for d in dims:
            n_elems *= d
        print(f"  n_elems={n_elems}")
        
        n_blocks = (n_elems + 255) // 256
        print(f"  n_blocks={n_blocks}")
        print(f"  Expected raw size (82 bytes/block): {n_blocks * 82}")
        
        # Seek to tensor data and read raw bytes
        f.seek(data_blob_offset + data_offset)
        
        # Read first 5 blocks
        for block_idx in range(5):
            raw = f.read(82)
            if len(raw) < 82:
                print(f"  Block {block_idx}: only got {len(raw)} bytes!")
                break
            
            # Parse block
            d_bytes = struct.unpack('<e', raw[0:2])[0]  # fp16
            qs = list(raw[2:66])  # 64 bytes
            qh = list(raw[66:74])  # 8 bytes
            scales = list(raw[74:82])  # 8 bytes
            
            print(f"\nBlock {block_idx}: d={d_bytes}")
            print(f"  qs[0..7]: {qs[0:8]}")
            print(f"  qh[{block_idx}]: {list(qh)}")
            print(f"  scales: {list(scales)}")
            
            # Dequantize first sub-block (ib32=0, l=0)
            db0 = d_bytes * (0.5 + (scales[0] & 0x0F)) * 0.25
            db1 = d_bytes * (0.5 + (scales[0] >> 4)) * 0.25
            print(f"  db[0]={db0}, db[1]={db1}")
            
            grid_idx_l0 = qs[0] | ((qh[0] << 8) & 0x300)
            print(f"  grid_idx(l=0): {grid_idx_l0}")
            
            # Load iq2s_grid to dequantize
            # The grid is 1024 uint64 entries, each packing 8 int8 values
            # We need to read the grid from the same file (it's embedded in the C code)
            # For now, just print the raw block data

if __name__ == '__main__':
    main()
