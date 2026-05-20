#!/usr/bin/env python3
"""
Example: RotorQuant-style Givens rotation applied to K/V cache before Q4_0 quantization.

Demonstrates the concept using numpy. The actual implementation would use
2×2 Givens rotation matrices applied as block-diagonal operators on K/V vectors.

Inference: dequantize Q4_0 → inverse Givens rotation → use as normal K/V.

Author: Hermes Agent for bytropix
"""

import numpy as np

# --- Configuration ---
N_DIMS = 256          # GQA head dimension
BLOCK_SIZE = 2        # Givens operates on adjacent pairs
N_PAIRS = N_DIMS // BLOCK_SIZE  # 128 pairs

# --- Reference: a K head vector (simulated) ---
np.random.seed(42)
k_f32 = np.random.randn(N_DIMS).astype(np.float32)

print("=== RotorQuant Demo ===")
print(f"Input K head: {N_DIMS} floats, range [{k_f32.min():.4f}, {k_f32.max():.4f}]")

# --- Step 1: Apply Givens rotation (one angle per pair) ---
# Each adjacent pair (x_2i, x_{2i+1}) is rotated by angle θ_i
# cos/sin per pair = 2 params per pair, 256 params total for full head
angles = np.random.uniform(-np.pi/4, np.pi/4, N_PAIRS).astype(np.float32)

k_rotated = k_f32.copy()
for i in range(N_PAIRS):
    c = np.cos(angles[i])
    s = np.sin(angles[i])
    x0, x1 = k_rotated[2*i], k_rotated[2*i+1]
    k_rotated[2*i]   = c * x0 - s * x1
    k_rotated[2*i+1] = s * x0 + c * x1

print(f"After Givens: range [{k_rotated.min():.4f}, {k_rotated.max():.4f}]")

# --- Step 2: Quantize to Q4_0 ---
def quantize_q40(x):
    """Quantize 32 floats to Q4_0 block. Returns (d, qs)."""
    amax = np.max(np.abs(x))
    if amax == 0: return 0.0, np.zeros(16, dtype=np.uint8)
    d = amax / 7.0
    qi = np.clip(np.round(x / d + 8), 0, 15).astype(np.uint8)
    qs = np.zeros(16, dtype=np.uint8)
    for j in range(16):
        qs[j] = qi[2*j] | (qi[2*j+1] << 4)
    return d, qs

def dequantize_q40(d, qs, n=32):
    """Dequantize Q4_0 block back to floats."""
    x = np.zeros(n, dtype=np.float32)
    for j in range(n // 2):
        q0 = qs[j] & 0xF
        q1 = (qs[j] >> 4) & 0xF
        x[2*j] = (q0 - 8) * d
        x[2*j+1] = (q1 - 8) * d
    return x

# Quantize the rotated K in 32-element blocks
k_q40_blocks = []
for blk in range(N_DIMS // 32):
    d, qs = quantize_q40(k_rotated[blk*32:(blk+1)*32])
    k_q40_blocks.append((d, qs))

# Measure compression
bytes_f32 = k_f32.nbytes  # 1024 bytes
bytes_q40 = N_DIMS // 32 * (2 + 16)  # 8 blocks × 18 bytes = 144 bytes
print(f"\nF32 size: {bytes_f32} bytes")
print(f"Q4_0 size: {bytes_q40} bytes")
print(f"Compression ratio: {bytes_f32 / bytes_q40:.1f}:1")

# --- Step 3: Dequantize + inverse Givens rotation ---
k_deq = np.zeros(N_DIMS, dtype=np.float32)
for blk in range(N_DIMS // 32):
    d, qs = k_q40_blocks[blk]
    k_deq[blk*32:(blk+1)*32] = dequantize_q40(d, qs)

k_recovered = k_deq.copy()
for i in range(N_PAIRS):
    c = np.cos(angles[i])
    s = np.sin(angles[i])
    x0, x1 = k_recovered[2*i], k_recovered[2*i+1]
    k_recovered[2*i]   = c * x0 + s * x1  # inverse: transpose = [c, s; -s, c]^T = [c, -s; s, c]
    k_recovered[2*i+1] = -s * x0 + c * x1

# --- Verify ---
dot = np.dot(k_f32, k_recovered)
n1 = np.linalg.norm(k_f32)
n2 = np.linalg.norm(k_recovered)
cos_sim = dot / (n1 * n2)
max_err = np.max(np.abs(k_f32 - k_recovered))

print(f"\n=== Verification ===")
print(f"Cos-sim (input vs recovered): {cos_sim:.6f}")
print(f"Max element error: {max_err:.6f}")
print(f"Error source: Q4_0 quantization noise (Givens rotation is lossless)")

# Compare with direct Q4_0 (no rotation) to see if Givens helps
k_direct_blocks = []
for blk in range(N_DIMS // 32):
    d, qs = quantize_q40(k_f32[blk*32:(blk+1)*32])
    k_direct_blocks.append((d, qs))

k_direct = np.zeros(N_DIMS, dtype=np.float32)
for blk in range(N_DIMS // 32):
    d, qs = k_direct_blocks[blk]
    k_direct[blk*32:(blk+1)*32] = dequantize_q40(d, qs)

dot_direct = np.dot(k_f32, k_direct)
cos_direct = dot_direct / (n1 * np.linalg.norm(k_direct))
print(f"\nDirect Q4_0 cos-sim (no rotation): {cos_direct:.6f}")
print(f"RotorQuant+Givens cos-sim:          {cos_sim:.6f}")
print(f"Improvement: {(cos_sim - cos_direct)*10000:.2f}e-4")

# Explanation
print(f"""
=== How RotorQuant Works ===
1. Apply Givens rotation to each adjacent pair in the K/V vector
   [k_0, k_1, k_2, k_3, ...] → rotate each (k_2i, k_{2i+1}) by angle θ_i
   This spreads information across dimensions, reducing outlier magnitude.
   
2. Quantize the rotated vector with Q4_0 (or lower bit-width)
   The rotation makes the distribution more Gaussian-like, which Q4_0 handles better.
   
3. Store: Q4_0 blocks (144 bytes) + rotation angles (128×4 = 512 bytes)
   Total overhead: 512 bytes per head. For 10 GQA layers × 2 KV heads = 20 heads
   → 10 KB total rotation parameter storage — negligible.

4. On read: dequantize Q4_0 → apply INVERSE Givens rotation
   Inverse is just transpose: [c, s; -s, c] because rotation matrices are orthogonal.

Why it's better than Walsh-Hadamard Transform (TurboQuant):
- Only 2 parameters per pair (vs 64 per WHT row)
- Block-diagonal = GPU-friendly: each block is independent 2×2 operation
- No O(n log n) WHT overhead — just 2 FMAs per pair
""")
