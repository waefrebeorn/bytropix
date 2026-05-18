#!/usr/bin/env python3
"""Compare our C RMSNorm vs Python RMSNorm for the SAME input."""
import numpy as np

# Load C embedding
c_emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32,
                    count=2048, offset=248044*2048*4)

# Load C normed output
c_normed = np.fromfile('/tmp/c_normed.bin', dtype=np.float32)

# Compute RMSNorm in Python
import gguf
r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
attn_norm = np.array([t for t in r.tensors if t.name == 'blk.0.attn_norm.weight'][0].data, dtype=np.float32)

def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return x / rms * w

py_normed = rms_norm(c_emb, attn_norm)

print(f"C emb first 10: {[f'{x:.8f}' for x in c_emb[:10]]}")
print(f"attn_norm first 10: {[f'{x:.8f}' for x in attn_norm[:10]]}")
print(f"C normed first 10: {[f'{x:.8f}' for x in c_normed[:10]]}")
print(f"Py normed first 10: {[f'{x:.8f}' for x in py_normed[:10]]}")

diff = np.abs(py_normed - c_normed)
print(f"\nMax diff: {diff.max():.10f}")
print(f"Mean diff: {diff.mean():.10f}")

# If they match, the RMSNorm is correct
if diff.max() < 1e-5:
    print("✓ RMSNorm MATCHES!")
else:
    print("✗ RMSNorm MISMATCH!")
    # Find where they differ
    bad = np.where(diff > 1e-5)[0]
    print(f"  First {min(10, len(bad))} bad indices: {bad[:10]}")
    for i in bad[:5]:
        print(f"  [{i}] C={c_normed[i]:.10f} Py={py_normed[i]:.10f} diff={diff[i]:.10f}")
