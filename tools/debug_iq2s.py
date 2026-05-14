#!/usr/bin/env python3
"""Dump raw bytes of first IQ2_S tensor and dequantize blocks manually."""
import struct
import sys

GRID_PATH = '/home/wubu/bytropix/src/iq2s_grid.inc'

def load_iq2s_grid():
    grid = []
    with open(GRID_PATH) as f:
        for line in f.read().split(','):
            line = line.strip()
            if line.startswith('0x') or line.startswith('0X'):
                grid.append(int(line, 0))
    return grid

def fp16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x03FF
    if exp == 0:
        return 0.0
    f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    return struct.unpack('>f', struct.pack('>I', f32))[0]

if __name__ == '__main__':
    path = sys.argv[1]
    iq2s_grid = load_iq2s_grid()
    print(f"Loaded iq2s_grid: {len(iq2s_grid)} entries")
    
    with open(path, 'rb') as f:
        f.read(4)  # magic
        struct.unpack('<I', f.read(4))[0]  # version
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensors: {n_tensors}, KV: {n_kv}")
        
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
        
        # Read tensor info
        target = None
        for i in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode()
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack('<I', f.read(4))[0]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            if ggml_type == 18 and target is None:
                target = (name, dims, ggml_type, data_offset, i)
        
        if not target:
            print("No IQ2_S tensor found")
            sys.exit(1)
        
        tensor_end = f.tell()
        alignment = 128
        pad = (alignment - (tensor_end % alignment)) % alignment
        data_blob_offset = tensor_end + pad
        
        name, dims, ggml_type, data_offset, idx = target
        n_elems = 1
        for d in dims:
            n_elems *= d
        n_blocks = (n_elems + 255) // 256
        
        print(f"\nFirst IQ2_S tensor [#{idx}]: {name}")
        print(f"  dims={dims} n_elems={n_elems} n_blocks={n_blocks}")
        print(f"  data_offset={data_offset} data_blob_offset={data_blob_offset}")
        
        # Seek to tensor data
        f.seek(data_blob_offset + data_offset)
        
        # Read and analyze first 5 blocks
        for block_idx in range(5):
            raw = f.read(82)
            if len(raw) < 82:
                print(f"  Block {block_idx}: only got {len(raw)} bytes!")
                break
            
            d_bits = struct.unpack('<H', raw[0:2])[0]
            d = fp16_to_f32(d_bits)
            qs = list(raw[2:66])
            qh = list(raw[66:74])
            scales = list(raw[74:82])
            
            print(f"\nBlock {block_idx}: d=0x{d_bits:04x} = {d:.8f}")
            print(f"  qs[0:8]    = {[f'0x{q:02x}' for q in qs[0:8]]}")
            print(f"  qh         = {[f'0x{q:02x}' for q in qh]}")
            print(f"  scales     = {[f'0x{s:02x}' for s in scales]}")
            
            # Dequant stats for entire block
            output = []
            for ib32 in range(8):
                ib_qs = qs[ib32*4 : ib32*4 + 4]
                ib_signs = qs[32 + ib32*4 : 32 + ib32*4 + 4]
                db0 = d * (0.5 + (scales[ib32] & 0x0F)) * 0.25
                db1 = d * (0.5 + (scales[ib32] >> 4)) * 0.25
                
                for l in range(4):
                    dl = db0 if l < 2 else db1
                    grid_idx = ib_qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)
                    
                    if grid_idx < len(iq2s_grid):
                        packed = iq2s_grid[grid_idx]
                        for j in range(8):
                            b = (packed >> (j * 8)) & 0xFF
                            if b >= 128: b -= 256
                            v = dl * b
                            if ib_signs[l] & (1 << j):
                                v = -v
                            output.append(v)
                    else:
                        output.extend([0.0] * 8)
            
            vmin = min(output)
            vmax = max(output)
            mean = sum(output) / len(output)
            var = sum((v - mean)**2 for v in output) / len(output)
            std = var ** 0.5
            extreme = sum(1 for v in output if abs(v) > 10)
            
            print(f"  Stats: min={vmin:.3f} max={vmax:.3f} mean={mean:.3f} std={std:.3f} |v|>10={extreme}/256")
            print(f"  First 8: {[f'{v:.4f}' for v in output[:8]]}")
