#!/usr/bin/env python3
"""Map SSM vs GQA layers."""
import numpy as np, os

ref_dir = "/tmp/ref_int"

print("=== Layer type classification ===")
print(f"{'Layer':>6} {'conv_in':>9} {'linear_attn':>12} {'attn_out':>9} {'final_out':>10} {'TYPE':>8}")
print("-" * 60)
for li in range(40):
    has_conv = os.path.exists(os.path.join(ref_dir, f"L{li}_conv_input.bin"))
    has_linear_attn = os.path.exists(os.path.join(ref_dir, f"L{li}_linear_attn_out.bin"))
    has_attn = os.path.exists(os.path.join(ref_dir, f"L{li}_attn_output.bin"))
    has_final = os.path.exists(os.path.join(ref_dir, f"L{li}_final_output.bin"))
    
    conv_std = np.fromfile(os.path.join(ref_dir, f"L{li}_conv_input.bin"), dtype=np.float32).std() if has_conv else 0
    linear_std = np.fromfile(os.path.join(ref_dir, f"L{li}_linear_attn_out.bin"), dtype=np.float32).std() if has_linear_attn else 0
    attn_std = np.fromfile(os.path.join(ref_dir, f"L{li}_attn_output.bin"), dtype=np.float32).std() if has_attn else 0
    final_std = np.fromfile(os.path.join(ref_dir, f"L{li}_final_output.bin"), dtype=np.float32).std() if has_final else 0
    
    attn_active = attn_std > 0.01
    
    # Classify
    if has_conv and attn_active:
        t = "BOTH"
    elif has_conv and not attn_active:
        t = "SSM-ONLY"
    elif not has_conv and attn_active:
        t = "GQA-ONLY"
    else:
        t = "?"
    
    # Check if this layer has the typical "3 SSM + 1 GQA" pattern
    three_ssm_group = li % 4  # 0,1,2 = SSM, 3 = GQA
    
    ci = "Y" if has_conv else "N"
    la = "Y" if has_linear_attn else "N"
    
    print(f"  L{li:02d}   {ci:>9} {la:>12} {attn_std:>9.4f} {final_std:>10.4f}  {t:>8} (group={three_ssm_group})")
