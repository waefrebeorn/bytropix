#!/usr/bin/env python3
"""Create a synthetic GGUF file with MXFP4 and NVFP4 tensors for testing
the dequant pipeline end-to-end.

Uses the GGUF format as expected by our C reader:
  - magic: b'gguf' (4 bytes)
  - version: uint32 LE (2)
  - tensor_count: uint64 LE
  - kv_count: uint64 LE
  - tensors: each has:
    key_len: uint64 LE, key bytes
    n_dims:  uint32 LE
    dims:    uint64 LE × n_dims
    ggml_type: uint32 LE
    data_offset: uint64 LE
  - kv entries: key_len (u64) + key + type (i32) + value
"""
import struct
import sys

GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_F32  = 0

def write_gguf(path):
    alignment = 32

    # --- MXFP4 tensor: 64 elements (2 blocks of 32) = 34 bytes ---
    # Block 0: scale=0x80 (2^(128-127)=2.0), values: 0x32 0x10
    #   nibble 3=2.0, nibble 2=1.0 → ×2 = 4, 2
    #   nibble 1=0.5, nibble 0=0.0 → ×2 = 1, 0
    mxfp4_data = bytearray(17 * 2)  # 34 bytes
    mxfp4_data[0]  = 0x80  # E8M0 scale byte
    mxfp4_data[1]  = 0x32  # hi=3(2.0), lo=2(1.0) → ×2 = [4, 2]
    mxfp4_data[2]  = 0x10  # hi=1(0.5), lo=0(0.0) → ×2 = [1, 0]
    mxfp4_data[17] = 0x82  # E8M0 scale byte: 2^(130-127)=8.0
    mxfp4_data[18] = 0x22  # hi=2(1.0), lo=2(1.0) → ×8 = [8, 8]
    mxfp4_data[19] = 0x11  # hi=1(0.5), lo=1(0.5) → ×8 = [4, 4]
    # Elements: [4,2,1,0, 0,0,..., 0,0, 8,8,4,4, 0,0...]

    # --- NVFP4 tensor: 64 elements (1 block of 64) = 36 bytes ---
    # 4 UE4M3 scale bytes + 32 bytes of packed E2M1
    nvfp4_data = bytearray(36)
    nvfp4_data[0] = 0x78  # UE4M3 scale: E=(0x78>>3)&0x1F=15, M=0x78&7=0 → (1+0/8)*2^(15-7)=256
    # Scale 1,2,3 are 0x00 → 0.0
    nvfp4_data[4] = 0x32  # E2M1: hi=3(2.0), lo=2(1.0) → ×256 = [512, 256]
    nvfp4_data[5] = 0x10  # E2M1: hi=1(0.5), lo=0(0.0) → ×256 = [128, 0]
    # Element 4+ are 0 (from zeroed qs)

    # --- F32 control tensor: 4 elements ---
    f32_data = struct.pack('<4f', 1.5, 2.5, 3.5, 4.5)

    tensors = [
        ('mxfp4_test.weight', [64], GGML_TYPE_MXFP4, bytes(mxfp4_data)),
        ('nvfp4_test.weight', [64], GGML_TYPE_NVFP4, bytes(nvfp4_data)),
        ('f32_test.weight',     [4],  GGML_TYPE_F32,   f32_data),
    ]

    # KV entries: general.architecture = "deepseek_v4"
    kv_data = bytearray()
    kv_key = b'general.architecture'
    kv_data += struct.pack('<Q', len(kv_key))  # key_len
    kv_data += kv_key
    kv_data += struct.pack('<i', 8)  # type = 8 (string)
    kv_data += struct.pack('<Q', len(b'deepseek_v4'))
    kv_data += b'deepseek_v4'

    n_kv = 1

    # Build tensor metadata section
    tensor_section = bytearray()
    for name, shape, gtype, data in tensors:
        name_bytes = name.encode('utf-8')
        tensor_section += struct.pack('<Q', len(name_bytes))  # key_len
        tensor_section += name_bytes
        tensor_section += struct.pack('<I', len(shape))       # n_dims
        for d in shape:                                        # dims (uint64 LE)
            tensor_section += struct.pack('<Q', d)
        tensor_section += struct.pack('<i', gtype)             # ggml_type
        tensor_section += struct.pack('<Q', 0)                 # data_offset (placeholder)

    # Header
    header = bytearray()
    header += b'GGUF'
    header += struct.pack('<I', 2)  # version 2
    header += struct.pack('<Q', len(tensors))  # tensor count
    header += struct.pack('<Q', n_kv)  # kv count

    # Calculate data region start (aligned)
    meta_size = len(header) + len(tensor_section) + len(kv_data)
    data_start = (meta_size + alignment - 1) & ~(alignment - 1)

    # Now fill in correct data_offsets
    # GGUF data_offset is relative to the data blob start (data_blob_offset)
    offset = 0  # relative offset within the data region
    entry_pos = 0  # running position within tensor_section
    for i, (name, shape, gtype, data) in enumerate(tensors):
        name_bytes = name.encode('utf-8')
        entry_size = 8 + len(name_bytes) + 4 + len(shape) * 8 + 4 + 8  # full entry size
        doff = entry_pos + 8 + len(name_bytes) + 4 + len(shape) * 8 + 4  # absolute
        struct.pack_into('<Q', tensor_section, doff, offset)
        entry_pos += entry_size
        offset += len(data)
        offset = (offset + alignment - 1) & ~(alignment - 1)

    # Write file
    with open(path, 'wb') as f:
        f.write(header)
        f.write(kv_data)  # GGUF spec: KV after tensors, but our C reader expects KV before tensor info
        f.write(tensor_section)
        # Pad to data_start
        pad = data_start - len(header) - len(tensor_section) - len(kv_data)
        f.write(b'\x00' * pad)
        # Write tensor data
        for name, shape, gtype, data in tensors:
            f.write(data)
            pad = (alignment - (len(data) % alignment)) % alignment
            f.write(b'\x00' * pad)

    print(f"Created synthetic GGUF: {path}")
    print(f"  Tensors: {len(tensors)}")
    for name, shape, gtype, data in tensors:
        print(f"    {name}: shape={shape} type={gtype} bytes={len(data)}")
    print(f"  Data starts at offset: {data_start}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_synthetic.gguf'
    write_gguf(path)
