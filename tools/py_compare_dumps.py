#!/usr/bin/env python3
"""Compare our layer dumps vs reference layer dumps."""
import numpy as np
import glob, os

D_MODEL = 2048

# Our layer dumps
our_in = {}
our_out = {}
for f in sorted(glob.glob("/tmp/dump_layers/our_layer_*_in.bin")):
    parts = os.path.basename(f).split('_')
    n = int(parts[2])
    our_in[n] = np.frombuffer(open(f, 'rb').read(), dtype=np.float32).reshape(-1, D_MODEL)
for f in sorted(glob.glob("/tmp/dump_layers/our_layer_*_out.bin")):
    parts = os.path.basename(f).split('_')
    n = int(parts[2])
    our_out[n] = np.frombuffer(open(f, 'rb').read(), dtype=np.float32).reshape(-1, D_MODEL)

# Reference dumps (ref_layer_N.bin)
ref = {}
for f in sorted(glob.glob("/tmp/dump_ref/ref_layer_*.bin")):
    parts = os.path.basename(f).replace('.bin','').split('_')
    n = int(parts[2])
    ref[n] = np.frombuffer(open(f, 'rb').read(), dtype=np.float32).reshape(-1, D_MODEL)

print(f"Our dumps: {len(our_in)} layers in, {len(our_out)} layers out")
print(f"Ref dumps: {len(ref)} layers (layers: {sorted(ref.keys())[:5]}...{sorted(ref.keys())[-5:]})")

# Get available common layers
common = sorted(set(our_out.keys()) & set(ref.keys()))
print(f"\nCommon layers: {len(common)}")

print("\n=== Per-layer comparison (our_out vs ref) ===")
for L in common:
    o = our_out[L].flatten()
    r = ref[L].flatten()
    
    dot = np.dot(o, r)
    n_o = np.linalg.norm(o)
    n_r = np.linalg.norm(r)
    cos = dot / (n_o * n_r + 1e-30)
    
    diff = o - r
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    
    # Compare our_in at layer L vs ref[L-1]
    if L > 0 and L in our_in:
        o_in = our_in[L].flatten()
        if L-1 in ref:
            r_prev = ref[L-1].flatten()
            cos_in = np.dot(o_in, r_prev) / (np.linalg.norm(o_in) * np.linalg.norm(r_prev) + 1e-30)
        else:
            cos_in = 0
    else:
        cos_in = 0
    
    print(f"  L{L:2d}: cos={cos:.6f}  max|diff|={max_abs:.6f}  mean|diff|={mean_abs:.6f}  (in_vs_ref_prev_cos={cos_in:.6f})")

print("\n=== Stats per layer ===")
for L in common[:10]:
    o = our_out[L].flatten()
    r = ref[L].flatten()
    print(f"  L{L:2d}: our_std={np.std(o):.4f} our_mean={np.mean(o):.4f}  ref_std={np.std(r):.4f} ref_mean={np.mean(r):.4f}")
