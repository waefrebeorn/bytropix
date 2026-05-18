#!/usr/bin/env python3
"""Compare specific token logits between our output and reference."""
import numpy as np

our = np.fromfile('/tmp/our_logits_40l.bin', dtype=np.float32)
ref = np.fromfile('/tmp/ref_logits.bin', dtype=np.float32)

print(f"Our logits range: {our.min():.4f} to {our.max():.4f}")
print(f"Ref logits range: {ref.min():.4f} to {ref.max():.4f}")

# Check specific known tokens from reference top-5
tokens = [220, 264, 2880, 11762, 279, 84944, 55073]
for t in tokens:
    if t < 50000:
        print(f"  Token {t}: our={our[t]:.4f} ref={ref[t]:.4f}")

# Check tokens from our top-5
our_tokens = [84944, 55073, 29521, 242813, 43540]
for t in our_tokens:
    val = our[t] if t < 50000 else float('nan')
    rval = ref[t] if t < 50000 else (float('nan') if t >= 50000 else 0)
    print(f"  Token {t}: our={our[t] if t < 50000 else 'N/A':.4f} ref={rval if isinstance(rval, float) else 'N/A'}")

# Also check what the top values are in our output for tokens 0-50000
top5_our = np.argsort(-our[:50000])[:10]
print("\nOur top-10 in first 50000:")
for t in top5_our:
    print(f"  [{t}]: our={our[t]:.4f} ref={ref[t]:.4f}")

top5_ref = np.argsort(-ref[:50000])[:10]
print("\nRef top-10 in first 50000:")
for t in top5_ref:
    print(f"  [{t}]: our={our[t]:.4f} ref={ref[t]:.4f}")

# Cos-sim per segment to see where it diverges
print("\nSegment cos-sim (first 500):")
for start in range(0, 50000, 500):
    end = min(start + 500, 50000)
    o = our[start:end]; r = ref[start:end]
    dot = np.dot(o, r)
    no = np.sqrt(np.dot(o, o)); nr = np.sqrt(np.dot(r, r))
    cs = dot / (no * nr + 1e-30)
    if abs(cs) < 0.5:  # Only show low cos-sim regions
        print(f"  [{start}:{end}] cos-sim={cs:.4f}")
