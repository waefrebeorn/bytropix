#!/usr/bin/env python3
"""Analyze per-layer hidden state evolution vs reference."""
import numpy as np
import os

ref = np.fromfile('/tmp/llama_embd.bin', dtype=np.float32)
print(f"Reference final hidden: std={ref.std():.4f}")

layer_dir = '/tmp/dump_layers'
files = sorted([f for f in os.listdir(layer_dir) if f.endswith('.bin')])

# Get all layers' outputs
for il in range(40):
    f_in = os.path.join(layer_dir, f'our_layer_{il}_in.bin')
    f_out = os.path.join(layer_dir, f'our_layer_{il}_out.bin')
    
    if not os.path.exists(f_in) or not os.path.exists(f_out):
        continue
    
    h_in = np.fromfile(f_in, dtype=np.float32)
    h_out = np.fromfile(f_out, dtype=np.float32)
    
    # Our "after layer l" state IS h_out (it's the residual after the layer)
    # Compare vs reference final
    cos_in = np.dot(h_in, ref) / (np.linalg.norm(h_in) * np.linalg.norm(ref))
    cos_out = np.dot(h_out, ref) / (np.linalg.norm(h_out) * np.linalg.norm(ref))
    
    # Also compare h_in vs h_out (how much did the layer change the state)
    layer_norm = np.linalg.norm(h_out - h_in)
    
    # Compare each dimension of h_out vs ref
    diff = np.abs(h_out - ref)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # Track reference-like dimensions
    # Check: does the hidden state REFERENCE's top dimensions exist in our h_out?
    top_ref_dims = np.argsort(np.abs(ref))[-10:]
    our_at_top = h_out[top_ref_dims]
    ref_at_top = ref[top_ref_dims]
    
    print(f"L{il:2d}: |Δlayer|={layer_norm:.4f}  "
          f"cos(in,ref)={cos_in:.4f}  cos(out,ref)={cos_out:.4f}  "
          f"max_diff={max_diff:.2f}  ref_dim_0_val={ref_at_top[0]:.2f} our_val={our_at_top[0]:.2f}  "
          f"h_out_std={h_out.std():.4f}")
