#!/usr/bin/env python3
"""Analyze per-layer hidden states from our model."""
import numpy as np
import os

for l in range(40):
    f_in = f'/tmp/dump_layers/our_layer_{l}_in.bin'
    f_out = f'/tmp/dump_layers/our_layer_{l}_out.bin'
    if not os.path.exists(f_in) or not os.path.exists(f_out):
        continue
    
    x_in = np.fromfile(f_in, dtype=np.float32)
    x_out = np.fromfile(f_out, dtype=np.float32)
    delta = x_out - x_in
    
    is_ssm = (l + 1) % 4 != 0
    layer_type = "SSM" if is_ssm else "GQA"
    
    in_mean, in_std = x_in.mean(), x_in.std()
    out_mean, out_std = x_out.mean(), x_out.std()
    delta_mean, delta_std = delta.mean(), delta.std()
    delta_norm = np.linalg.norm(delta)
    in_norm = np.linalg.norm(x_in)
    rel = delta_norm / (in_norm + 1e-30)
    
    print(f"L{l:2d} {layer_type:3s}: in_std={in_std:.4f} out_std={out_std:.4f} "
          f"Δstd={delta_std:.4f} rel={rel:.4f} delta_max={np.max(np.abs(delta)):.4f}")
