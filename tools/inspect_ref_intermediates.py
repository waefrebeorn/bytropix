#!/usr/bin/env python3
"""Inspect reference intermediate tensor dumps from llama.cpp."""
import numpy as np
import os, sys, glob

ref_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref_intermediates"

files = sorted(glob.glob(os.path.join(ref_dir, "L0_*.bin")))

print(f"=== Layer 0 Intermediates ({len(files)} files) ===\n")

# Group by architecture step
groups = {
    "norm": [],
    "conv": [],
    "gated_delta": [],
    "ssm_recurrence": [],
    "output_proj": [],
    "attn": [],
    "ffn_moe": [],
    "ffn_shared": [],
    "final": [],
}

for f in files:
    name = os.path.basename(f)
    d = np.fromfile(f, dtype=np.float32)
    # Classify
    if "norm" in name:
        groups["norm"].append((name, d))
    elif "conv" in name:
        groups["conv"].append((name, d))
    elif "beta" in name or "alpha" in name or "gate" in name or "delta" in name or "predelta" in name:
        groups["gated_delta"].append((name, d))
    elif "state" in name or "linear_attn" in name or "k_conv" in name or "q_conv" in name or "v_conv" in name:
        groups["ssm_recurrence"].append((name, d))
    elif "attn" in name:
        groups["attn"].append((name, d))
    elif "ffn_moe" in name or "moe_logits" in name or "moe_probs" in name or "moe_gate" in name or "moe_down" in name:
        groups["ffn_moe"].append((name, d))
    elif "shexp" in name or "shared_expert" in name or "ffn_up" in name or "ffn_gate" in name or "ffn_swiglu" in name:
        groups["ffn_shared"].append((name, d))
    elif "l_out" in name or "final_output" in name or "post_moe" in name:
        groups["final"].append((name, d))
    else:
        groups["final"].append((name, d))

for gname, tensor_list in groups.items():
    if not tensor_list:
        continue
    print(f"--- {gname} ---")
    for name, d in tensor_list:
        # Try to infer shape
        if d.size == 2048:
            shape_str = "[2048]"
        elif d.size == 8192:
            shape_str = "[8192]"
        elif d.size == 4096:
            shape_str = "[4096]"
        elif d.size % 2048 == 0:
            n_tokens = d.size // 2048
            shape_str = f"[{n_tokens}, 2048]"
        elif d.size % 8192 == 0:
            n_tokens = d.size // 8192
            shape_str = f"[{n_tokens}, 8192]"
        elif d.size % 128 == 0 and d.size > 10000:
            n_tokens = d.size // 128
            shape_str = f"[{n_tokens}, 128]"
        else:
            shape_str = f"[{d.size}]"
        print(f"  {name}: size={d.size:8d} shape={shape_str} min={d.min():8.6f} max={d.max():8.6f} mean={d.mean():8.6f} std={d.std():8.6f}")
    print()
