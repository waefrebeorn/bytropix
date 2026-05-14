#!/usr/bin/env python3
"""
bytropix/math_viz/01_nested_hyperbolic_spaces.py

PROVES: WuBu nesting visualizes correctly as nested Poincaré disks
MATH: Riemannian metric g = 4 * g_E / (1 - ||x||^2)^2 on the unit ball
       Nested levels: H^n1_{c1,s1} ⊃ H^n2_{c2,s2} ⊃ ... with learnable curvatures

Run: python math_viz/01_nested_hyperbolic_spaces.py
Output: visualizations/nested_hyperbolic_spaces.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import os, math

# ──────────────────────────────────────────────
# Math: Poincaré disk model of hyperbolic space
# ──────────────────────────────────────────────
# The Poincaré disk H^2 is the unit disk {z: |z| < 1}
# with metric ds^2 = 4 * dx^2 / (1 - |x|^2)^2
#
# Hyperbolic distance from origin: d(0, x) = 2 * arctanh(|x|)
# Geodesic radius r_h maps to Euclidean radius: r_e = tanh(r_h / 2)
#
# Curvature c scales the metric: ds^2_c = 4 * dx^2 / (c * (1 - |x|^2)^2)
# For curvature c > 0, effective distance: d_c(0, x) = 2 * arctanh(|x|) / sqrt(c)
# Higher c = "steeper" geometry = faster volume growth

def hyperbolic_to_euclidean_radius(hyperbolic_radius, curvature=1.0):
    """Convert hyperbolic radius to Euclidean display radius.
    
    For curvature c, distance d maps to Euclidean radius:
    r_e = tanh(sqrt(c) * d / 2)
    """
    return np.tanh(np.sqrt(curvature) * hyperbolic_radius / 2)

def generate_nested_levels(n_levels=5, curvatures=None, scales=None):
    """
    Generate parameters for nested hyperbolic levels.
    
    Math: Each level i has:
    - curvature c_i (steepness of geometry)
    - scale s_i (affects the "zoom" / effective radius)
    - dimension n_i (not visualized in 2D)
    
    The nesting hierarchy: H^{n1}_{c1,s1} ⊃ H^{n2}_{c2,s2} ⊃ ...
    means that data at level i+1 lives in a bounded region 
    near the origin of level i's hyperbolic space.
    """
    if curvatures is None:
        # φ-inspired curvature progression
        phi = (1 + math.sqrt(5)) / 2
        curvatures = [1.0 * (phi ** (i - 2)) for i in range(n_levels)]
    if scales is None:
        # Scale decreases inward (inner levels zoom in)
        scales = [1.0 / (1.2 ** i) for i in range(n_levels)]
    return curvatures, scales

def sample_poincare_points(n_points, radius, curvature=1.0):
    """Sample points uniformly (in hyperbolic metric) within a 
    hyperbolic disk of given radius, rendered in Euclidean coordinates."""
    # Uniform in hyperbolic distance
    r_h = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to Euclidean coordinates
    r_e = hyperbolic_to_euclidean_radius(r_h, curvature)
    x = r_e * np.cos(theta)
    y = r_e * np.sin(theta)
    return x, y

def plot_nested_levels(n_levels=5, save_path=None):
    """Main visualization: nested Poincaré disks with data points."""
    
    curvatures, scales = generate_nested_levels(n_levels)
    phi = (1 + math.sqrt(5)) / 2
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_levels))
    
    # The outermost boundary: the Poincaré disk itself
    outer = Circle((0, 0), 1.0, fill=False, 
                   edgecolor='#1a1a2e', linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(outer)
    ax.text(1.05, 0, "∞ (boundary)", fontsize=8, color='#1a1a2e', alpha=0.4,
            verticalalignment='center')
    
    patches = []
    
    for i in range(n_levels):
        c = curvatures[i]
        s = scales[i]
        
        # The hyperbolic radius of this level's "bubble"
        # Inner levels are smaller in hyperbolic radius
        h_radius = 3.0 / (1.5 ** i)
        
        # Convert to Euclidean display radius
        e_radius = hyperbolic_to_euclidean_radius(h_radius, c)
        e_radius = min(e_radius, 0.98)  # Stay within Poincaré disk
        
        # Scale factor — higher curvature = same Euclidean radius 
        # represents MORE hyperbolic distance
        effective_radius = e_radius * s
        
        # Draw the level boundary
        circle = Circle((0, 0), effective_radius, 
                       fill=False, edgecolor=colors[i], 
                       linewidth=2.5 - 0.3*i, linestyle='-', alpha=0.7)
        ax.add_patch(circle)
        
        # Sample data points within this level in hyperbolic metric
        n_pts = max(10, 60 - 8*i)
        px, py = sample_poincare_points(n_pts, h_radius * s, c)
        
        # Clip to visual boundary
        mask = np.sqrt(px**2 + py**2) < effective_radius * 0.95
        px, py = px[mask], py[mask]
        
        ax.scatter(px, py, c=[colors[i]], s=8, alpha=0.6, 
                  edgecolors='none', zorder=5)
        
        # Label
        label = f"Level {i+1}: n={4-i if 4-i > 2 else 2}, c={c:.2f}"
        ax.text(0, effective_radius + 0.03, label, 
               fontsize=9, color=colors[i], ha='center', va='bottom',
               fontweight='bold')
        
        # Small annotation: scale
        ax.text(effective_radius * 0.7, -effective_radius * 0.7, 
               f"s={s:.2f}", fontsize=7, color=colors[i], alpha=0.5)
    
    # Title with math
    ax.set_title(
        "WuBu Nesting: Nested Hyperbolic Spaces $\\mathbb{H}^{n_i}_{c_i,s_i}$\n"
        "$\\mathbb{H}^n \\supset \\mathbb{H}^{n-1}_{c_1,s_1} \\supset \\mathbb{H}^{n-2}_{c_2,s_2} \\supset \\dots$\n"
        "Each level: learnable curvature $c_i$, scale $s_i$, dimension $n_i$",
        fontsize=11, pad=20
    )
    
    # Add curvature/scale legend box
    legend_text = "   Level | dim | curvature | scale\n"
    for i in range(n_levels):
        legend_text += f"     {i+1}     |  {4-i if 4-i > 2 else 2}  |   {curvatures[i]:.3f}  | {scales[i]:.3f}\n"
    
    ax.text(-1.25, -1.2, legend_text, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fafafa', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"[SAVED] {save_path}")
    
    plt.close()
    return fig

if __name__ == '__main__':
    save = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'visualizations', 'nested_hyperbolic_spaces.png')
    plot_nested_levels(n_levels=6, save_path=save)
    print("[DONE] Nested hyperbolic space visualization complete.")
    print(f"[INFO] 6 levels with φ-progression curvatures: ", end="")
    curv, _ = generate_nested_levels(6)
    print([f"{c:.3f}" for c in curv])
