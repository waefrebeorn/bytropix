#!/usr/bin/env python3
"""
Example: Hamilton Encoder-style quaternion manifold compression for V cache.

This demonstrates the concept from the HASHMIND project: compress V cache
tokens into a 5D quaternion grid, use BSP tree for subset retrieval.

The actual project lives at:
  /mnt/c/projects/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/

Key architectural insight: Instead of quantizing individual values (Q4_0),
compress entire token representations into a learned manifold and only
materialize relevant subsets at decode time.

Author: Hermes Agent for bytropix
"""

import numpy as np

print("=== Hamilton Encoder Attention Concept ===")
print("""
Problem: At 256k context, GQA attention reads ALL K/V cache entries per decode.
  Even with Q4_0 (4:1), we still read 720 MB per full attention pass.
  With sliding window 16384: 46 MB per pass — but lose long-range context.

Hamilton solution: Compress V cache into a quaternion manifold, use BSP tree
  for O(log N) retrieval. Only materialize ~64 tokens at decode time.
""")

# --- Simulate V cache with 1000 tokens ---
np.random.seed(42)
N_TOKENS = 1000
D_MODEL = 2048
D_V = 512  # GQA KV dim

v_cache = np.random.randn(N_TOKENS, D_V).astype(np.float32)
# Add structure: nearby tokens are similar (as in real text)
for i in range(1, N_TOKENS):
    v_cache[i] = 0.9 * v_cache[i-1] + 0.1 * v_cache[i]

print(f"V cache: {N_TOKENS} tokens × {D_V} dims = {v_cache.nbytes / 1024:.0f} KB")

# --- Step 1: MLP Encoder — compress V to 5D quaternion ---
# Simple 2-layer MLP: D_V → 32 → 5
np.random.seed(123)
W1 = np.random.randn(D_V, 32).astype(np.float32) / np.sqrt(D_V)
b1 = np.zeros(32, dtype=np.float32)
W2 = np.random.randn(32, 5).astype(np.float32) / np.sqrt(32)
b2 = np.zeros(5, dtype=np.float32)

def mlp_encode(v):
    h = np.maximum(0, v @ W1 + b1)  # ReLU
    return h @ W2 + b2  # 5D output: [qx, qy, qz, qw, magnitude]

v_enc = np.array([mlp_encode(v) for v in v_cache])
# Normalize quaternion part
for i in range(N_TOKENS):
    q_norm = np.linalg.norm(v_enc[i, :4])
    if q_norm > 0: v_enc[i, :4] /= q_norm

print(f"\nEncoded to 5D quaternion grid: {v_enc.nbytes / 1024:.1f} KB")
print(f"Compression ratio: {v_cache.nbytes / v_enc.nbytes:.0f}:1")

# --- Step 2: Build 2D grid from quaternion positions ---
# Each token's quaternion (qx, qy, qz, qw) determines its grid cell
# Project to 2D using qx, qy as spatial coordinates
GRID_SIZE = int(np.ceil(np.sqrt(N_TOKENS)))
grid = -np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int32)
cell_tokens = {}  # cell_id → list of token indices

for i in range(N_TOKENS):
    cx = int((v_enc[i, 0] * 0.5 + 0.5) * (GRID_SIZE - 1))
    cy = int((v_enc[i, 1] * 0.5 + 0.5) * (GRID_SIZE - 1))
    cx = np.clip(cx, 0, GRID_SIZE - 1)
    cy = np.clip(cy, 0, GRID_SIZE - 1)
    cell_id = cy * GRID_SIZE + cx
    if cell_id not in cell_tokens:
        cell_tokens[cell_id] = []
    cell_tokens[cell_id].append(i)

print(f"\nGrid: {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} cells")
print(f"Occupied cells: {len(cell_tokens)}/{GRID_SIZE*GRID_SIZE}")
print(f"Avg tokens/cell: {N_TOKENS/len(cell_tokens):.1f}")

# --- Step 3: Simulate a query — find nearest cell ---
query = np.random.randn(D_MODEL).astype(np.float32)
query_enc = mlp_encode(query)

# Find nearest cell by quaternion distance
best_cell = None
best_dist = float('inf')
for cell_id, tokens in cell_tokens.items():
    # Use the first token in cell as cell centroid (simplified)
    centroid = v_enc[tokens[0]]
    dist = np.linalg.norm(query_enc[:4] - centroid[:4])
    if dist < best_dist:
        best_dist = dist
        best_cell = cell_id

selected_tokens = cell_tokens.get(best_cell, [])
print(f"\n=== Query ===")
print(f"Nearest cell: {best_cell} (dist={best_dist:.4f})")
print(f"Tokens in cell: {len(selected_tokens)}")
print(f"Full attention would read: {N_TOKENS} tokens")
print(f"Subset attention reads:     {len(selected_tokens)} tokens")
print(f"Reduction: {N_TOKENS / max(1, len(selected_tokens)):.0f}×")

# --- Step 4: Full vs subset attention comparison ---
# Simplified: just compare Q·K scores for full vs subset
q_scores_full = np.dot(query[:256], v_cache[:, :256].T)
q_scores_subset = np.zeros_like(q_scores_full)
for t in selected_tokens:
    q_scores_subset[t] = np.dot(query[:256], v_cache[t, :256])

# Check if subset captures top-8
top8_full = np.argsort(q_scores_full)[-8:]
top8_subset = np.argsort(q_scores_subset)[-8:]
overlap = len(set(top8_full) & set(top8_subset))

print(f"\nTop-8 tokens from full attention: {top8_full}")
print(f"Top-8 tokens from subset attention: {top8_subset}")
print(f"Overlap: {overlap}/8")
print(f"Recall@8: {overlap/8*100:.0f}%")

print(f"""
=== Hamilton Encoder Architecture Summary ===
1. MLP Encoder: V_{D_V} → 32 → 5 (per token, O(N) per chunk)
   - 4 quaternion components (qx,qy,qz,qw) for spatial position on grid
   - 1 magnitude for salience (attention weight bias)

2. 2D Grid: SQRT(N)×SQRT(N) cells, tokens assigned by quaternion position
   - Tokens with similar content → nearby cells (manifold learning property)
   - Grid rebuild only when grid dimensions change (~every 256 tokens)

3. BSP Tree: O(log N) partition of grid cells for retrieval
   - Each query projects to quaternion space → BSP returns best cell
   - Only tokens in that cell are attended (N_g / N_cells tokens)

4. Subset Attention: Read Q4_0 dequantized K/V only for selected tokens
   - Combined with Q4_0: 4× × (N / N_cells) = 40-100× total reduction

5. The 4096 Recall Window Bug (HASHMIND legacy):
   - Cell indices were computed as `token_pos % 4096` instead of absolute
   - Fix: use absolute token position for cell indexing
   - Symptom: tokens beyond 4096 appeared in wrong cells → garbage attention
""")
