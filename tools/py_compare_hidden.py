#!/usr/bin/env python3
"""Compare reference hidden state vs our hidden state for BOS token."""
import numpy as np

# Reference hidden state (from llama.cpp with embeddings=true)
ref = np.fromfile('/tmp/llama_embd.bin', dtype=np.float32)
print(f"Reference hidden state: shape={ref.shape}, mean={ref.mean():.6f}, std={ref.std():.6f}")
print(f"ref[0..4]: {[f'{x:.8f}' for x in ref[:5]]}")

# Our hidden state (from the full model test)
# Let's find it - check if there are dump files
import os

# Check /tmp/our_hidden.bin (saved by our test)
our_paths = [
    '/tmp/our_hidden.bin',           # from test_full_model
    '/tmp/ref_hidden.bin',           # reference hidden
    '/tmp/dump_layers/our_layer_39_out.bin',  # last layer output
]
for p in our_paths:
    if os.path.exists(p):
        our = np.fromfile(p, dtype=np.float32)
        print(f"\n{os.path.basename(p)}: shape={our.shape}, mean={our.mean():.6f}, std={our.std():.6f}")
        print(f"our[0..4]: {[f'{x:.8f}' for x in our[:5]]}")
        
        # Compare
        if len(our) == len(ref):
            cos = np.dot(our, ref) / (np.linalg.norm(our) * np.linalg.norm(ref))
            print(f"\nCos-sim: {cos:.6f}")
            mse = np.mean((our - ref)**2)
            print(f"MSE: {mse:.10f}")
            
            # Check per-dimension differences
            diff = np.abs(our - ref)
            print(f"Max abs diff: {diff.max():.6f}")
            print(f"Mean abs diff: {diff.mean():.6f}")
            
            # Dimensions where difference > 0.1
            bad = np.where(diff > 0.1)[0]
            print(f"Dimensions with diff > 0.1: {len(bad)} / {len(ref)}")
            
            # Dimensions where reference has largest values
            top_ref_dims = np.argsort(np.abs(ref))[-10:]
            print(f"\nTop ref dimensions: {top_ref_dims}")
            print(f"Ref values at those: {ref[top_ref_dims]}")
            print(f"Our values at those: {our[top_ref_dims]}")
        else:
            print(f"Shape mismatch: ref={len(ref)}, our={len(our)}")

# Also check the dump_layers
layer_dir = '/tmp/dump_layers'
if os.path.exists(layer_dir):
    files = sorted([f for f in os.listdir(layer_dir) if f.endswith('.bin')])
    print(f"\nLayer dump files ({len(files)}):")
    for f in files:
        print(f"  {f}: {os.path.getsize(os.path.join(layer_dir, f))} bytes")
        if files.index(f) < 3:
            data = np.fromfile(os.path.join(layer_dir, f), dtype=np.float32)
            print(f"    mean={data.mean():.6f}, std={data.std():.6f}, first={data[0]:.4f}")
