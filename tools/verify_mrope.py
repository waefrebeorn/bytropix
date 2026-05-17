#!/usr/bin/env python3
"""Verify MRoPE frequency computation vs continuous progression."""
import numpy as np

ROPE_THETA = 10000000.0
ROTARY_DIM = 64

# Old continuous: pair i gets freq = base^(-2*i/64)
print("=== OLD (continuous) vs NEW (MRoPE sections [11,11,10]) ===")
print()

# Compare frequencies at position 1
pos = 1
sec_lengths = [11, 11, 10]
pair_count = ROTARY_DIM // 2  # 32

# Old: continuous progression
old_freqs = np.zeros(pair_count)
for i in range(pair_count):
    old_freqs[i] = ROPE_THETA ** (-2.0 * i / ROTARY_DIM)

# New: per-section restart
new_freqs = np.zeros(pair_count)
pair_idx = 0
for s in range(3):
    for i in range(sec_lengths[s]):
        new_freqs[pair_idx] = ROPE_THETA ** (-2.0 * i / ROTARY_DIM)
        pair_idx += 1

print("Pair | Old freq | New freq | Old theta | New theta")
print("-" * 60)
for i in range(pair_count):
    old_angle = pos * old_freqs[i]
    new_angle = pos * new_freqs[i]
    match = "SAME" if old_freqs[i] == new_freqs[i] else "DIFF"
    print(f"{i:3d}  | {old_freqs[i]:.6f} | {new_freqs[i]:.6f} | {old_angle:.6f} | {new_angle:.6f}  {match}")

# Show where they differ
diffs = np.where(np.abs(old_freqs - new_freqs) > 1e-10)[0]
print(f"\nDifferences at pairs: {diffs}")
print(f"Total different pairs: {len(diffs)} / {pair_count}")

# MRoPE key insight: pairs 0 and 11 both have freq=1.0 (base^0)
print(f"\nKey insight:")
print(f"  Pair 0: freq={new_freqs[0]:.10f} (base^0 = 1)")
print(f"  Pair 11: freq={new_freqs[11]:.10f} (base^0 = 1, section reset)")
print(f"  Old pair 11 had: freq={old_freqs[11]:.10f} (much lower)")

# Show the repeating pattern
print("\n=== MRoPE Frequency Pattern (by section) ===")
pair_idx = 0
for s in range(3):
    print(f"Section {s} ({sec_lengths[s]} pairs): ", end="")
    for i in range(sec_lengths[s]):
        freq = ROPE_THETA ** (-2.0 * i / ROTARY_DIM)
        print(f"{freq:.4f} ", end="")
    print()
