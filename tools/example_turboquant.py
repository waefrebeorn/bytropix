#!/usr/bin/env python3
"""
Example: TurboQuant-style Walsh-Hadamard Transform (WHT) + Q4_0 quantization.

WHT spreads information across dimensions before quantization, reducing
the impact of outlier values on quantization error.

Author: Hermes Agent for bytropix
"""

import numpy as np

def hadamard_matrix(n):
    """Generate Walsh-Hadamard matrix of order n (power of 2)."""
    H = np.array([[1]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

# --- Configuration ---
N_DIMS = 256          # GQA head dimension
N_PAIRS = N_DIMS // 2

# --- Reference: a V head vector (simulated — V tends to have more outliers) ---
np.random.seed(42)
v_f32 = np.random.randn(N_DIMS).astype(np.float32)
# Add some outliers (V cache often has extreme values)
v_f32[::16] *= 5.0

print("=== TurboQuant Demo ===")
print(f"Input V head: {N_DIMS} floats, range [{v_f32.min():.4f}, {v_f32.max():.4f}]")
print(f"  Outliers present at every 16th position")

# --- Step 1: Apply Walsh-Hadamard Transform ---
H = hadamard_matrix(N_DIMS)
v_wht = (H @ v_f32) / np.sqrt(N_DIMS)  # normalized WHT

print(f"\nAfter WHT: range [{v_wht.min():.4f}, {v_wht.max():.4f}]")
print(f"  Std before WHT: {v_f32.std():.4f}, after WHT: {v_wht.std():.4f}")
print(f"  WHT spreads outlier energy across all dimensions")

def quantize_q40(x):
    amax = np.max(np.abs(x))
    if amax == 0: return 0.0, np.zeros(16, dtype=np.uint8)
    d = amax / 7.0
    qi = np.clip(np.round(x / d + 8), 0, 15).astype(np.uint8)
    qs = np.zeros(16, dtype=np.uint8)
    for j in range(16):
        qs[j] = qi[2*j] | (qi[2*j+1] << 4)
    return d, qs

def dequantize_q40(d, qs, n=32):
    x = np.zeros(n, dtype=np.float32)
    for j in range(n // 2):
        q0 = qs[j] & 0xF
        q1 = (qs[j] >> 4) & 0xF
        x[2*j] = (q0 - 8) * d
        x[2*j+1] = (q1 - 8) * d
    return x

# --- Step 2: Quantize WHT output with Q4_0 ---
v_q40_blocks = []
for blk in range(N_DIMS // 32):
    d, qs = quantize_q40(v_wht[blk*32:(blk+1)*32])
    v_q40_blocks.append((d, qs))

# Compression
bytes_f32 = v_f32.nbytes
bytes_q40 = N_DIMS // 32 * 18
print(f"\nF32: {bytes_f32}B → Q4_0: {bytes_q40}B ({bytes_f32/bytes_q40:.1f}:1)")

# --- Step 3: Dequantize + inverse WHT ---
v_deq = np.zeros(N_DIMS, dtype=np.float32)
for blk in range(N_DIMS // 32):
    d, qs = v_q40_blocks[blk]
    v_deq[blk*32:(blk+1)*32] = dequantize_q40(d, qs)
v_recovered = (H @ v_deq) / np.sqrt(N_DIMS)  # inverse WHT

# --- Compare: direct Q4_0 without WHT ---
v_direct_blocks = []
for blk in range(N_DIMS // 32):
    d, qs = quantize_q40(v_f32[blk*32:(blk+1)*32])
    v_direct_blocks.append((d, qs))
v_direct = np.zeros(N_DIMS, dtype=np.float32)
for blk in range(N_DIMS // 32):
    d, qs = v_direct_blocks[blk]
    v_direct[blk*32:(blk+1)*32] = dequantize_q40(d, qs)

# --- Verify ---
cos_wht = np.dot(v_f32, v_recovered) / (np.linalg.norm(v_f32) * np.linalg.norm(v_recovered))
cos_direct = np.dot(v_f32, v_direct) / (np.linalg.norm(v_f32) * np.linalg.norm(v_direct))

print(f"\n=== Comparison ===")
print(f"Direct Q4_0 (no WHT):    cos-sim = {cos_direct:.6f}")
print(f"TurboQuant (WHT+Q4_0):   cos-sim = {cos_wht:.6f}")
print(f"Improvement: {(cos_wht - cos_direct)*10000:.2f}e-4")
print(f"Max error direct:  {np.max(np.abs(v_f32 - v_direct)):.6f}")
print(f"Max error WHT:     {np.max(np.abs(v_f32 - v_recovered)):.6f}")

print(f"""
=== How TurboQuant Works ===
1. Apply Walsh-Hadamard Transform (WHT) to K/V vector
   WHT = H @ x / sqrt(n) — orthogonal, spreads energy across all dims
   
2. Quantize with PolarQuant or Q4_0 on the transformed vector
   The transformed vector has more uniform distribution → less quantization error
   
3. Store: Q4_0 blocks + WHT metadata (none needed — H is fixed)
   Advantage over RotorQuant: no rotation parameters to store
   Disadvantage: O(n log n) WHT vs O(n) Givens per write

4. On read: dequantize Q4_0 → inverse WHT (= same as forward WHT)
   H^-1 = H for normalized WHT (H is symmetric orthogonal)

Why WHT helps with outliers:
  Without WHT: outlier at position 0 gets quantized to ±7*scale, error = ±scale
  With WHT: outlier energy spreads across all dims, each gets smaller error
""")
