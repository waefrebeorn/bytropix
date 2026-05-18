#!/usr/bin/env python3
"""Compare our per-layer hidden states vs reference per-layer hidden states."""
import numpy as np
import os

print("Layer-by-layer comparison: ours vs reference")
print("="*80)
print(f"{'L':>3} | {'Our_in_std':>10} | {'Our_out_std':>11} | {'Ref_out_std':>11} | {'Ref_last_std':>12} | {'cos(in,ref)':>10} | {'cos(out,ref)':>10} | {'cos(our,ref_layer)':>17} | {'max|diff|':>9}")
print("-"*80)

ref_final = np.fromfile('/tmp/llama_embd.bin', dtype=np.float32)

for il in range(40):
    # Our layer outputs
    our_in = np.fromfile(f'/tmp/dump_layers/our_layer_{il}_in.bin', dtype=np.float32)
    our_out = np.fromfile(f'/tmp/dump_layers/our_layer_{il}_out.bin', dtype=np.float32)
    
    # Reference layer outputs
    ref_out = np.fromfile(f'/tmp/dump_ref/ref_layer_{il}.bin', dtype=np.float32)
    
    # Comparison metrics
    cos_in_ref = np.dot(our_in, ref_out) / (np.linalg.norm(our_in) * np.linalg.norm(ref_out))
    cos_out_ref = np.dot(our_out, ref_out) / (np.linalg.norm(our_out) * np.linalg.norm(ref_out))
    cos_ourref = np.dot(our_out, ref_out) / (np.linalg.norm(our_out) * np.linalg.norm(ref_out))
    diff = np.abs(our_out - ref_out)
    max_diff = diff.max()
    
    # The reference ref layer output should include the residual
    # but our our_layer_x_in is the residual BEFORE layer x
    # and our_layer_x_out is the residual AFTER layer x
    # and ref_layer_x.bin is the layer OUTPUT before residual + residual
    
    # Actually, looking at qwen35moe.cpp - the dump is "cur" at the end of the layer
    # This is AFTER: cur = ggml_add(ctx0, cur, ffn_residual)
    # and BEFORE: inpL = cur
    # So it IS the hidden state AFTER the layer
    
    # Compare our out with ref (both are hidden state AFTER the layer)
    # For a fair comparison, plot the standard deviation
    print(f"{il:3d} | {our_in.std():10.4f} | {our_out.std():11.4f} | {ref_out.std():11.4f} | {our_out.std()-ref_out.std():12.4f} | {cos_in_ref:10.4f} | {cos_out_ref:10.4f} | {cos_ourref:17.4f} | {max_diff:9.2f}")
    
print("="*80)
# Focus on first layer
print("\n=== DETAILED LAYER 0 COMPARISON ===")
l0_our = np.fromfile('/tmp/dump_layers/our_layer_0_out.bin', dtype=np.float32)
l0_ref = np.fromfile('/tmp/dump_ref/ref_layer_0.bin', dtype=np.float32)

# Check if they have same magnitude
print(f"Our  L0 out: mean={l0_our.mean():.8f} std={l0_our.std():.8f}")
print(f"Ref  L0 out: mean={l0_ref.mean():.8f} std={l0_ref.std():.8f}")

# What's the ratio?
ratio = l0_our / np.where(l0_ref != 0, l0_ref, 1e-30)
print(f"Ratio: mean={np.mean(ratio[~np.isinf(ratio) & ~np.isnan(ratio)]):.4f} "
      f"std={np.std(ratio[~np.isinf(ratio) & ~np.isnan(ratio)]):.4f}")

# Check if the relationship is systematic
# Try sign match
sign_match = np.mean((l0_our > 0) == (l0_ref > 0))
print(f"Sign match rate: {sign_match:.4f}")

# Linear regression
from numpy.linalg import lstsq
A = l0_ref.reshape(-1, 1)
try:
    coeff, residuals, _, _ = lstsq(A, l0_our, rcond=None)
    print(f"Best linear fit: ours = {coeff[0]:.6f} * ref")
    print(f"Residual std: {np.sqrt(residuals[0]/len(l0_our)) if len(residuals) > 0 else 'N/A':.6f}")
except:
    print("Linear fit failed (rank deficiency)")

# Check correlation per dimension range
for start in range(0, 2048, 256):
    end = min(start+256, 2048)
    r = l0_our[start:end]
    f = l0_ref[start:end]
    if np.linalg.norm(r) > 0 and np.linalg.norm(f) > 0:
        c = np.dot(r, f) / (np.linalg.norm(r) * np.linalg.norm(f))
        print(f"  dims {start:4d}-{end-1:4d}: cos={c:.4f}")
