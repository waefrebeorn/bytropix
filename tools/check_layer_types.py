#!/usr/bin/env python3
"""Check which layers have active attention (GQA) vs zero attention (SSM)."""
import numpy as np, os

ref_dir = "/tmp/ref_int"

print("=== Attention activity per layer ===")
for li in range(40):
    fn = os.path.join(ref_dir, f"L{li}_attn_output.bin")
    if os.path.exists(fn):
        d = np.fromfile(fn, dtype=np.float32)
        is_zero = d.max() == 0.0
        tag = "ZERO (SSM)" if is_zero else "ACTIVE (GQA)"
        print(f"  L{li:02d}_attn_output: size={d.size:6d} std={d.std():.6f} max={d.max():.6f}  <- {tag}")
    else:
        # Check if the file might exist for this layer at all
        # (GQA layers 30-39 might not have attn_output)
        print(f"  L{li:02d}_attn_output: NOT FOUND")

# Also check which layers have conv_input (SSM-specific)
print("\n=== SSM conv_input activity ===")
for li in range(40):
    fn = os.path.join(ref_dir, f"L{li}_conv_input.bin")
    if os.path.exists(fn):
        d = np.fromfile(fn, dtype=np.float32)
        print(f"  L{li:02d}_conv_input: size={d.size:6d} std={d.std():.6f}  <- SSM LAYER")

# Check l_out for all layers  
print("\n=== Layer output (l_out) ===")
for li in range(40):
    fn = os.path.join(ref_dir, f"L{li}_l_out.bin")
    if os.path.exists(fn):
        d = np.fromfile(fn, dtype=np.float32)
        print(f"  L{li:02d}_l_out: size={d.size:6d} std={d.std():.6f}")
