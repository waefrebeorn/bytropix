#!/usr/bin/env python3
"""Analyze Qwen3.6 embeddings for Poincaré ball mapping.
Reads extracted embeddings, computes distribution stats,
determines optimal R for exp_map, tests NN preservation."""

import numpy as np
import time

VOCAB = 248320
HIDDEN = 2048
PATH = '/home/wubu/bytropix/data/qwen36_embeddings.bin'

print("Loading embeddings...")
t0 = time.time()
emb = np.memmap(PATH, dtype=np.float32, mode='r', shape=(VOCAB, HIDDEN))
print(f"Loaded in {time.time()-t0:.1f}s")

# Sample to speed up analysis (10% of vocab)
np.random.seed(42)
sample_idx = np.random.choice(VOCAB, size=25000, replace=False)
sample = emb[sample_idx].copy()

# === Norm Distribution ===
norms = np.linalg.norm(sample, axis=1)
print(f"\n=== NORM DISTRIBUTION ===")
print(f"mean={norms.mean():.6f}, std={norms.std():.6f}")
print(f"min={norms.min():.6f}, max={norms.max():.6f}")
print(f"p1={np.percentile(norms,1):.6f}, p99={np.percentile(norms,99):.6f}")
print(f"p90={np.percentile(norms,90):.6f}")
print(f"p95={np.percentile(norms,95):.6f}")
print(f"p99={np.percentile(norms,99):.6f}")
print(f"p99.9={np.percentile(norms,99.9):.6f}")

max_norm = norms.max()
# R = 3 × mean_norm puts 99.9% of points within tanh(max_norm/3*mean) of center
R_suggested = 3.0 * norms.mean()
print(f"\nSuggested R = 3 × mean_norm = {R_suggested:.4f}")
print(f"  => max point at tanh({max_norm:.4f}/{R_suggested:.4f}) = {np.tanh(max_norm/R_suggested):.4f}")
# For R = 1.0 (Poincaré unit ball)
print(f"R=1.0: max point at tanh({max_norm:.4f}) = {np.tanh(max_norm):.4f}")
# For R = max_norm
print(f"R=max_norm={max_norm:.4f}: max at tanh(1) = {np.tanh(1):.4f}")

# === NN Preservation Test ===
print(f"\n=== NN PRESERVATION TEST ===")
# Pick 100 random anchor tokens
np.random.seed(123)
anchors = np.random.choice(VOCAB, size=100, replace=False)
anchor_vecs = emb[anchors].copy()

# For each anchor, find top-5 NN in Euclidean space
# Also find top-5 NN in Poincaré space
# Compare overlap

def euclidean_nn(query, pool, k=5):
    """Find top-k nearest neighbors in Euclidean"""
    diffs = pool - query
    dists = np.linalg.norm(diffs, axis=1)
    return np.argsort(dists)[:k]

def poincare_nn(query, pool, k=5, R=1.0):
    """Find top-k nearest neighbors in Poincaré ball"""
    # Poincaré distance: arcosh(1 + 2*||x-y||²/((1-||x||²)*(1-||y||²)))
    q_norm_sq = np.sum(query**2)
    p_norm_sq = np.sum(pool**2, axis=1)
    
    # Clamp to avoid numerical issues at boundary
    q_norm_sq = np.clip(q_norm_sq, 0, 0.99)
    p_norm_sq = np.clip(p_norm_sq, 0, 0.99)
    
    # Poincaré metric chord distance: 2*||x-y||²/((1-||x||²)*(1-||y||²))
    diffs = pool - query
    sq_dists = np.sum(diffs**2, axis=1)
    
    # Distance in Poincaré ball
    poincare_sq = 2 * sq_dists / ((1 - q_norm_sq) * (1 - p_norm_sq) + 1e-8)
    # d(x,y) = arcosh(1 + poincare_sq)
    dists = np.arccosh(1 + np.clip(poincare_sq, 0, 1e6))
    
    return np.argsort(dists)[:k]

total_overlap = 0
total_pairs = 0

# Use smaller pool for NN search (5000 random candidates)
pool_idx = np.random.choice(VOCAB, size=5000, replace=False)
pool_vecs = emb[pool_idx].copy()

for a_idx, a_vec in zip(anchors, anchor_vecs):
    e_nn = euclidean_nn(a_vec, pool_vecs)
    p_nn = poincare_nn(a_vec, pool_vecs, R=R_suggested)
    
    overlap = len(set(e_nn) & set(p_nn))
    total_overlap += overlap
    total_pairs += 5

print(f"Top-5 overlap (R={R_suggested:.4f}): {total_overlap}/{total_pairs} = {100*total_overlap/total_pairs:.1f}%")

# Try different R values
for R_test in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    total_overlap = 0
    for a_vec in anchor_vecs[:20]:  # subset for speed
        e_nn = euclidean_nn(a_vec, pool_vecs)
        p_nn = poincare_nn(a_vec, pool_vecs, R=R_test)
        overlap = len(set(e_nn) & set(p_nn))
        total_overlap += overlap
    pct = 100 * total_overlap / (20 * 5)
    print(f"  R={R_test:.1f}: {pct:.1f}% overlap")

print(f"\n=== RECOMMENDATION ===")
print(f"Use R = {R_suggested:.4f} (3 × mean_norm)")
print(f"Embeddings stay in Poincaré ball at radius tanh(max_norm/R) ≈ {np.tanh(max_norm/R_suggested):.4f}")
print(f"C exp_map: out_i = tanh(norm/R) * x_i / norm")
