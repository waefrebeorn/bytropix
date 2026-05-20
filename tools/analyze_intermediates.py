#!/usr/bin/env python3
"""Analyze intermediate tensor dumps from DUMP_INTERMEDIATE_DIR."""
import numpy as np
import os, sys

ref_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref_int"

# Key intermediates to check per layer
key_names = ['conv_input', 'conv_output_silu', 'linear_attn_out', 'attn_output', 
             'ffn_moe_out', 'l_out', 'beta_sigmoid', 'a_softplus', 'gate', 'new_state',
             'state_predelta', 'final_output', 'norm', 'attn_residual']

print("=== Layer-specific intermediates ===")
for li in [0, 1, 29, 30, 39]:
    for kn in key_names:
        fn = os.path.join(ref_dir, f'L{li}_{kn}.bin')
        if os.path.exists(fn):
            d = np.fromfile(fn, dtype=np.float32)
            # Infer shape
            if d.size == 2048:
                shape = "[2048]"
            elif d.size == 32:
                shape = "[32]"
            elif d.size == 8192:
                shape = "[8192]"
            elif d.size == 4096:
                shape = "[4096]"
            elif d.size == 24576:
                shape = "[12, 2048]"
            elif d.size % 2048 == 0:
                ntok = d.size // 2048
                shape = f"[{ntok}, 2048]"
            else:
                shape = f"[{d.size}]"
            print(f"  L{li}_{kn}: size={d.size:8d} shape={shape} min={d.min():8.4f} max={d.max():8.4f} mean={d.mean():8.6f} std={d.std():8.6f}")
    print()

# Count unique intermediate names per layer
print("=== Unique intermediate names for L0 ===")
names = set()
for fn in os.listdir(ref_dir):
    if fn.startswith('L0_'):
        # Extract name part (remove L0_ prefix)
        name = fn[3:]
        # Remove .bin suffix
        name = name.rsplit('.', 1)[0]
        names.add(name)
for n in sorted(names):
    print(f"  {n}")

print(f"\nTotal unique L0 names: {len(names)}")
