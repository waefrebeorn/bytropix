#!/usr/bin/env python3
"""Verify BOS embedding used by our model == what llama.cpp uses."""
import numpy as np
import gguf

# Get BOS embedding from GGUF directly
r = gguf.GGUFReader('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]

bos_id = 248044
bos_embd = np.array(t.data[bos_id, :], dtype=np.float32)
print(f"BOS embedding (direct from GGUF):")
print(f"  shape={bos_embd.shape}, mean={bos_embd.mean():.6f}, std={bos_embd.std():.6f}")
print(f"  [0..4]: {[f'{x:.8f}' for x in bos_embd[:5]]}")

# Compare with our embedding file
our_emb = np.fromfile('data/qwen36_embeddings_c.bin.raw', dtype=np.float32)
our_bos = our_emb[bos_id * 2048 : (bos_id + 1) * 2048]
print(f"\nOur BOS embedding (from our file):")
print(f"  [0..4]: {[f'{x:.8f}' for x in our_bos[:5]]}")

diff = np.abs(bos_embd - our_bos)
print(f"\n  Max diff: {diff.max():.10f}")
print(f"  Mean diff: {diff.mean():.10f}")
if diff.max() < 1e-5:
    print("  ✅ EXACT MATCH!")
else:
    print("  ❌ MISMATCH!")
    bad = np.where(diff > 1e-5)[0]
    print(f"  {len(bad)} mismatching elements")
    for i in bad[:3]:
        print(f"  [{i}]: gguf={bos_embd[i]:.10f} our={our_bos[i]:.10f}")

# Also get the reference hidden state
ref = np.fromfile('/tmp/llama_embd.bin', dtype=np.float32)
print(f"\nReference final hidden (after all layers + output_norm):")
print(f"  [0..4]: {[f'{x:.8f}' for x in ref[:5]]}")

# Our final hidden from test_full_model (no MoE)
our = np.fromfile('/tmp/our_hidden.bin', dtype=np.float32)
print(f"\nOur final hidden:")
print(f"  [0..4]: {[f'{x:.8f}' for x in our[:5]]}")

# Our layer 0 output (with MoE-disabled passthrough)
l0_out = np.fromfile('/tmp/dump_layers/our_layer_0_out.bin', dtype=np.float32)
print(f"\nOur layer 0 output:")
print(f"  std={l0_out.std():.4f}, [0..4]: {[f'{x:.8f}' for x in l0_out[:5]]}")

# The reference final hidden state after output_norm is 2.5x larger std than our layer 0
# Since the reference layers accumulate over 40 layers, layer 0 alone won't match
# But we can check if the layer 0 output has the right SIGN for key dimensions
print(f"\nRef top-5 dims: {np.argsort(np.abs(ref))[-5:]}")
print(f"  ref[613]={ref[613]:.4f} our[layer0,613]={l0_out[613]:.4f}")

# Check the SSM layer 0 contribution direction
# If the layer contribution is in the same direction as reference for some dims
contribution = l0_out - np.fromfile('/tmp/dump_layers/our_layer_0_in.bin', dtype=np.float32)
print(f"\nOur layer 0 contribution (out - in):")
print(f"  std={contribution.std():.4f}")
# Is this in the same direction as the overall reference final state?
cos = np.dot(contribution, ref) / (np.linalg.norm(contribution) * np.linalg.norm(ref))
print(f"  cos(ref, contribution)={cos:.4f}")
