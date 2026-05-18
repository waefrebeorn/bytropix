#!/usr/bin/env python3
import gguf
import numpy as np

r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'output.weight'][0]
raw = np.array(t.data).view(np.uint8).flatten()

def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        val = (mant / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        val = float('inf') if mant == 0 else float('nan')
    else:
        val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val

D = 2048
out = np.zeros(D, dtype=np.float32)

for b in range(8):
    off = b * 144
    d = f16_to_f32(int.from_bytes(bytes(raw[off:off+2]), 'little'))
    dmin = f16_to_f32(int.from_bytes(bytes(raw[off+2:off+4]), 'little'))
    scales = [raw[off+4+i] for i in range(12)]
    qs = [raw[off+16+i] for i in range(128)]
    
    for g in range(0, 8, 2):
        j_g = g
        if j_g < 4:
            sc = scales[j_g] & 63
            m = scales[j_g + 4] & 63
        else:
            sc = (scales[j_g+4] & 0xF) | ((scales[j_g-4] >> 6) << 4)
            m = (scales[j_g+4] >> 4) | ((scales[j_g-0] >> 6) << 4)
        d1 = float(d) * sc
        m1 = float(dmin) * m
        
        j_g2 = g+1
        if j_g2 < 4:
            sc2 = scales[j_g2] & 63
            m2 = scales[j_g2 + 4] & 63
        else:
            sc2 = (scales[j_g2+4] & 0xF) | ((scales[j_g2-4] >> 6) << 4)
            m2 = (scales[j_g2+4] >> 4) | ((scales[j_g2-0] >> 6) << 4)
        d2_val = float(d) * sc2
        m2v = float(dmin) * m2
        
        qsg = qs[g//2 * 32 : (g//2 + 1) * 32]
        jj = (g // 2) * 64
        for l in range(32):
            idx = b * 256 + jj + l
            if idx < D:
                out[idx] = d1 * (qsg[l] & 0xF) - m1
            idx = b * 256 + jj + 32 + l
            if idx < D:
                out[idx] = d2_val * (qsg[l] >> 4) - m2v

print(f"First 10: {[f'{x:.8f}' for x in out[:10]]}")
print(f"Mean={float(out.mean()):.8f} Std={float(out.std()):.8f}")
out.tofile('/tmp/py_outw_token0.bin')
print("Saved OK")
