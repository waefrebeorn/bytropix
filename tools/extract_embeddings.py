#!/usr/bin/env python3
"""
Extract token_embd.weight from Qwen3.6 GGUF (Q5_K quantized).
Q5_K: 256 elements per block, 176 bytes per block.
Each block: 2B f16 d + 2B f16 dmin + 12B scales + 32B high bits + 128B low 4 bits.
Sub-blocks of 32: 5-bit values = low 4 bits + 1 high bit.
"""

import struct, sys, os, numpy as np, time

GGML_TYPE_Q5_K = 13
QK_K = 256
BLOCK_SIZE = 176  # sizeof(block_q5_K) = 2+2+12+32+128

def f16_to_f32(bits):
    """Convert uint16 (f16 bits) to float32"""
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x03FF
    if exp == 0:
        return 0.0  # simplified
    return (1 - 2*sign) * (2**(exp - 15)) * (1 + mant/1024)

def dequantize_q5_K_block(block_bytes):
    """Dequantize a single Q5_K block (176 bytes → 256 float32 values)"""
    # Parse block
    d = f16_to_f32(struct.unpack('<H', block_bytes[0:2])[0])
    dmin = f16_to_f32(struct.unpack('<H', block_bytes[2:4])[0])
    scales = block_bytes[4:16]      # 12 bytes, 6-bit values
    qh = block_bytes[16:48]          # 32 bytes, high bits
    qs = block_bytes[48:176]         # 128 bytes, low 4 bits
    
    result = np.zeros(256, dtype=np.float32)
    
    # 4 groups of 64, each group has 2 sub-blocks of 32
    is_idx = 0
    u1, u2 = 1, 2
    for group in range(4):  # 4 groups × 64 = 256
        # Get scale and min for first 32-element sub-block
        sc1 = scales[is_idx] & 0x3F
        m1 = (scales[is_idx] >> 4) & 0x3F  # wait, this is wrong
        # Actually scales are packed as 6-bit pairs in 12 bytes
        # Let me use get_scale_min_k4 logic
        
        # For now, unpack manually using numpy patterns
        sc_arr = np.frombuffer(scales, dtype=np.uint8)
        # Decode 6-bit values from 12 bytes (16 values)
        scaled = np.zeros(16, dtype=np.uint8)
        for i in range(12):
            if i < 8:
                scaled[i] = sc_arr[i] & 0x3F
                scaled[8+i] = (sc_arr[i] >> 6) | ((sc_arr[i+4] & 0x0F) << 2) if i < 4 else sc_arr[i] >> 6
            ...
        
        is_idx += 2
        u1 <<= 2
        u2 <<= 2
    
    return result

def extract_embeddings(gguf_path, out_path):
    t0 = time.time()
    f = open(gguf_path, 'rb')
    
    # Header
    assert f.read(4) == b'GGUF', "Not GGUF"
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<q', f.read(8))[0]
    n_kv = struct.unpack('<q', f.read(8))[0]
    print(f"GGUF v{ver}, {n_tensors} tensors, {n_kv} KV pairs")
    
    alignment = 32
    
    # Parse KV pairs
    for i in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        typ = struct.unpack('<i', f.read(4))[0]
        
        if typ == 8:
            vlen = struct.unpack('<Q', f.read(8))[0]
            f.seek(vlen, 1)
            val = None
        elif typ == 4:
            val = struct.unpack('<I', f.read(4))[0]
        elif typ == 5:
            val = struct.unpack('<i', f.read(4))[0]
        elif typ == 6:
            val = struct.unpack('<f', f.read(4))[0]
        elif typ == 7:
            val = struct.unpack('<?', f.read(1))[0]
        elif typ == 10:
            val = struct.unpack('<Q', f.read(8))[0]
        elif typ == 11:
            val = struct.unpack('<q', f.read(8))[0]
        elif typ == 9:
            arr_type = struct.unpack('<i', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:
                for _ in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, 1)
            else:
                elem_sizes = {0:1,1:1,7:1,2:2,3:2,4:4,5:4,10:8,11:8,12:8}
                f.seek(arr_len * elem_sizes.get(arr_type, 4), 1)
            val = None
        else:
            val = None
            f.seek(4, 1)  # skip unknown
        
        if key in ('general.alignment',):
            alignment = val
        elif key in ('qwen35moe.embedding_length', 'qwen35moe.block_count',
                     'qwen35moe.context_length', 'qwen35moe.expert_count'):
            print(f"  {key} = {val}")
    
    # Read tensor info
    tensors = []
    for i in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack('<i', f.read(4))[0]
        data_offset = struct.unpack('<Q', f.read(8))[0]
        tensors.append((name, dims, ggml_type, data_offset))
    
    data_start = f.tell()
    pad = (alignment - (data_start % alignment)) % alignment
    data_blob = data_start + pad
    
    # Find token_embd.weight
    target = None
    for name, dims, ggml_type, data_offset in tensors:
        if name == 'token_embd.weight':
            target = (name, dims, ggml_type, data_offset)
            break
    
    if not target:
        print("ERROR: token_embd.weight not found!")
        return False
    
    name, dims, ggml_type, data_offset = target
    hidden_f, vocab = dims
    print(f"\nFound '{name}': [{vocab}, {hidden_f}], type={ggml_type}")
    assert ggml_type == GGML_TYPE_Q5_K, f"Expected Q5_K (13), got {ggml_type}"
    
    # Calculate raw size
    n_total = hidden_f * vocab
    n_blocks_total = (n_total + QK_K - 1) // QK_K
    raw_size = n_blocks_total * BLOCK_SIZE
    print(f"Total elements: {n_total:,}, blocks: {n_blocks_total:,}")
    print(f"Raw data: {raw_size:,} bytes ({raw_size/1e6:.1f}MB)")
    
    # Read raw data
    f.seek(data_blob + data_offset)
    raw_bytes = f.read(raw_size)
    f.close()
    
    # Dequantize using vectorized numpy
    print("Dequantizing Q5_K...")
    tt = time.time()
    
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, BLOCK_SIZE)
    n_blocks = raw.shape[0]
    
    # Extract d and dmin (float16 at bytes 0-3)
    d_bits = raw[:, 0:2].view(np.uint16).ravel()
    dmin_bits = raw[:, 2:4].view(np.uint16).ravel()
    
    # f16 → f32 for d and dmin
    def hf_to_f32(bits):
        s = (bits >> 15) & 1
        e = (bits >> 10) & 0x1F
        m = bits & 0x03FF
        return np.where(e == 0, 0.0,
                        (1.0 - 2.0 * s.astype(np.float32)) *
                        (2.0 ** (e.astype(np.float32) - 15.0)) *
                        (1.0 + m.astype(np.float32) / 1024.0))
    
    d_vals = hf_to_f32(d_bits)   # [n_blocks]
    dmin_vals = hf_to_f32(dmin_bits)  # [n_blocks]
    
    # Scales: 12 bytes per block, packed as 16 × 6-bit values
    # Each byte stores two 6-bit values in get_scale_min_k4 format
    scales_raw = raw[:, 4:16]  # [n_blocks, 12]
    
    # Decode get_scale_min_k4: each 6-bit value split across bytes
    # Standard format: 16 values from 12 bytes
    # Each position i=0..15 maps to:
    #   sc[i] = (scales_raw[i//2] >> (i%2 ? 6 : 0)) & 0x3F  for i < 8
    #   sc[i] = ((scales_raw[i-8] >> 2) & 0x30) | (scales_raw[i-4] >> 6) ???
    # This is complex. Let me use a different approach:
    
    # Extract scales by iterating sub-blocks following C code
    result = np.zeros(n_blocks * QK_K, dtype=np.float32)
    
    qs_bytes = raw[:, 16:144]  # [n_blocks, 128] low 4 bits
    qh_bytes = raw[:, 144:176] # [n_blocks, 32] high 1 bits
    scales_raw = raw[:, 4:16]   # [n_blocks, 12]
    
    for b in range(n_blocks):
        d = d_vals[b]
        dmin = dmin_vals[b]
        ql = qs_bytes[b]
        qhh = qh_bytes[b]
        sc = scales_raw[b]
        
        offset = b * QK_K
        
        is_idx = 0
        u1, u2 = 1, 2
        for j in range(0, QK_K, 64):
            # get_scale_min_k4: extract from byte pairs
            # scales are packed as: byte[is_idx] = (m1<<4) | sc1, byte[is_idx+1] = (m2<<4) | sc2
            b1 = sc[is_idx]
            sc1 = b1 & 0x0F
            m1 = (b1 >> 4) & 0x0F
            
            b2 = sc[is_idx + 1]
            sc2 = b2 & 0x0F
            m2 = (b2 >> 4) & 0x0F
            
            d1 = d * sc1
            m1_val = dmin * m1
            d2 = d * sc2
            m2_val = dmin * m2
            
            ql_base = j // 2  # 32 bytes per 64 elements
            
            for l in range(32):
                idx1 = offset + j + l
                idx2 = offset + j + 32 + l
                
                lo_byte = ql[ql_base + l]
                hi_byte = qhh[l]
                
                val1 = (lo_byte & 0x0F) + (16 if (hi_byte & u1) else 0)
                val2 = (lo_byte >> 4) + (16 if (hi_byte & u2) else 0)
                
                result[idx1] = d1 * val1 - m1_val
                result[idx2] = d2 * val2 - m2_val
            
            is_idx += 2
            u1 <<= 2
            u2 <<= 2
    
    print(f"  Dequant time: {time.time()-tt:.1f}s")
    
    # Reshape: [hidden, vocab] → [vocab, hidden]
    embeddings = result.reshape(hidden_f, vocab).T.astype(np.float32)
    
    print(f"Shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norms: mean={norms.mean():.4f}, std={norms.std():.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")
    
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embeddings.tofile(out_path)
    with open(out_path + '.meta', 'w') as mf:
        mf.write(f"shape {vocab} {hidden_f}\nsource {gguf_path}\ntensor {name}\ndtype float32\n")
        mf.write(f"norm_mean {norms.mean():.6f}\nnorm_std {norms.std():.6f}\n")
        mf.write(f"norm_min {norms.min():.6f}\nnorm_max {norms.max():.6f}\n")
    
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"\nSaved: {out_path} ({size_gb:.2f}GB)")
    print(f"Within 15GB limit: {'YES' if size_gb < 15 else 'NO!'}")
    print(f"Total: {time.time()-t0:.1f}s")
    return True

if __name__ == '__main__':
    gguf = sys.argv[1] if len(sys.argv) > 1 else '/mnt/wslg/distro/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf'
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/wubu/bytropix/data/qwen36_embeddings.bin'
    if not os.path.exists(gguf):
        print(f"Not found: {gguf}")
        sys.exit(1)
    sys.exit(0 if extract_embeddings(gguf, out) else 1)
