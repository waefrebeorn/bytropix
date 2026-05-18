#!/usr/bin/env python3
"""Compare our no-MoE vs reference no-MoE layer-by-layer."""
import numpy as np
import os

print("="*80)
print("NO-MOE COMPARISON: Ours vs Reference (LLAMA_NO_MOE=1)")
print("="*80)

# Final hidden states
ref_final = np.fromfile('/tmp/dump_no_moe/ref_layer_39.bin', dtype=np.float32)
print(f"Reference final (no-MoE): std={ref_final.std():.4f}, first={ref_final[0]:.6f}")

# Our final (last layer output)
our_final = np.fromfile('/tmp/dump_layers/our_layer_39_out.bin', dtype=np.float32)
print(f"Our final (no-MoE):      std={our_final.std():.4f}, first={our_final[0]:.6f}")

cos = np.dot(our_final, ref_final) / (np.linalg.norm(our_final) * np.linalg.norm(ref_final))
print(f"Final cos-sim: {cos:.6f}")

print("\nPer-layer comparison:")
print(f"{'L':>3} | {'Our_std':>8} | {'Ref_std':>8} | {'cos(our,ref)':>12} | {'max|diff|':>9} | {'|Δour|':>7} | {'|Δref|':>7}")
print("-"*70)

for il in range(40):
    our = np.fromfile(f'/tmp/dump_layers/our_layer_{il}_out.bin', dtype=np.float32)
    ref = np.fromfile(f'/tmp/dump_no_moe/ref_layer_{il}.bin', dtype=np.float32)
    
    c = np.dot(our, ref) / (np.linalg.norm(our) * np.linalg.norm(ref))
    diff = np.abs(our - ref)
    
    # Compute delta from previous layer
    if il > 0:
        our_prev = np.fromfile(f'/tmp/dump_layers/our_layer_{il-1}_out.bin', dtype=np.float32)
        ref_prev = np.fromfile(f'/tmp/dump_no_moe/ref_layer_{il-1}.bin', dtype=np.float32)
        delta_our = np.linalg.norm(our - our_prev)
        delta_ref = np.linalg.norm(ref - ref_prev)
    else:
        our_prev = np.fromfile(f'/tmp/dump_layers/our_layer_0_in.bin', dtype=np.float32)
        ref_prev = np.zeros(2048, dtype=np.float32)
        delta_our = np.linalg.norm(our - our_prev)
        delta_ref = np.linalg.norm(ref)
    
    print(f"{il:3d} | {our.std():8.4f} | {ref.std():8.4f} | {c:12.4f} | {diff.max():9.4f} | {delta_our:7.2f} | {delta_ref:7.2f}")

# Now separate: how much of the divergence is from SSM layers vs GQA layers
print("\n=== LAYER TYPE ANALYSIS ===")
for il in [0, 1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39]:
    our = np.fromfile(f'/tmp/dump_layers/our_layer_{il}_out.bin', dtype=np.float32)
    ref = np.fromfile(f'/tmp/dump_no_moe/ref_layer_{il}.bin', dtype=np.float32)
    ltype = "GQA" if il % 4 == 3 else "SSM"
    
    # What fraction of the final hidden state is explained by the reference?
    c = np.dot(our, ref) / (np.linalg.norm(our) * np.linalg.norm(ref))
    
    print(f"Layer {il:2d} ({ltype}): cos={c:.4f}, our_std={our.std():.4f}, ref_std={ref.std():.4f}")
