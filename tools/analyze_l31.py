#!/usr/bin/env python3
"""Analyze L31 intermediate tensor shapes for GQA attention comparison."""
import numpy as np, os

ref_dir = "/tmp/ref_int"

print("=== L31 Attention intermediates ===")
for f in sorted(os.listdir(ref_dir)):
    if f.startswith("L31_"):
        d = np.fromfile(os.path.join(ref_dir, f), dtype=np.float32)
        # Infer shape
        if d.size == 20480:
            shape = "[10, 2048]"
        elif d.size == 10240:
            shape = "[5, 2048]"
        elif d.size == 8192:
            shape = "[8192]"
        elif d.size == 4096:
            shape = "[4096]"
        elif d.size == 5120:
            shape = "[10, 512] or [5, 1024]"
        elif d.size == 2048:
            shape = "[2048]"
        elif d.size == 40960:
            shape = "[20, 2048]"
        elif d.size == 2560:
            shape = "[5, 512]"
        elif d.size == 320:
            shape = "[5, 64] or [10, 32]"
        elif d.size == 160:
            shape = "[5, 32] or [10, 16]"
        elif d.size % 248320 == 0:
            shape = "[248320]"
        else:
            # Try to find shape by D_MODEL, q_dim, kv_dim factors
            shape = f"[{d.size}]"
        print(f"  {f}: {d.size:>8d} {shape:>20s} min={d.min():8.4f} max={d.max():8.4f} mean={d.mean():.6f} std={d.std():.6f}")
