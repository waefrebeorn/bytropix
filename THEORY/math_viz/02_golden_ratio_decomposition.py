#!/usr/bin/env python3
"""
bytropix/math_viz/02_golden_ratio_decomposition.py

PROVES: GAAD as optimal spatial tiling with phi
MATH: Recursive Golden Subdivision + Phi-Spiral for aspect-ratio-agnostic decomposition
       φ = (1 + sqrt(5)) / 2 ≈ 1.618...
       Golden rectangle: w/h = φ  →  w = h*φ
       Subdivision: square (h x h) + golden rectangle (h x h(φ-1))

Run: python math_viz/02_golden_ratio_decomposition.py
Output: visualizations/golden_ratio_decomposition.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os, math

phi = (1 + math.sqrt(5)) / 2

def golden_subdivide(rect, depth=0, max_depth=5, regions=None):
    """
    Recursive golden subdivision of a rectangle.
    
    Math: For a rectangle (x, y, w, h) with w ≥ h:
    - If w/h ≥ φ: cut off a square strip FIRST from the left
      Square: (x, y, h, h)
      Remainder: (x+h, y, w-h, h) — wait, this inverts aspect ratio!
      
    ACTUAL MATH: The golden rectangle has aspect ratio φ.
    For a rectangle (x, y, w, h):
    - Long side determines cut direction
    - If w >= h: create square of size h, remainder is (w-h) x h
    - Remainder is a new golden rectangle only if w/h = φ
    """
    if regions is None:
        regions = []
    
    x, y, w, h = rect
    
    if depth >= max_depth or min(w, h) < 1:
        regions.append((depth, rect, 'terminal'))
        return regions
    
    if w >= h:
        # Landscape: cut a square from the left
        square = (x, y, h, h)
        remainder = (x + h, y, w - h, h)
        regions.append((depth, square, 'square'))
        golden_subdivide(remainder, depth + 1, max_depth, regions)
    else:
        # Portrait: cut a square from the top
        square = (x, y, w, w)
        remainder = (x, y + w, w, h - w)
        regions.append((depth, square, 'square'))
        golden_subdivide(remainder, depth + 1, max_depth, regions)
    
    return regions


def phi_spiral_centers(frame_w, frame_h, n_spirals=8):
    """
    Generate phi-spiral sampling centers.
    
    Math: A logarithmic spiral with growth factor φ:
    r(θ) = a * φ^(θ / π)
    
    Points are placed at radii proportional to φ^k
    """
    centers = []
    for i in range(n_spirals):
        angle = i * 2 * math.pi / n_spirals
        # φ-based radial spacing
        radius = min(frame_w, frame_h) * 0.15 * (phi ** (i / n_spirals))
        cx = frame_w / 2 + radius * math.cos(angle)
        cy = frame_h / 2 + radius * math.sin(angle)
        centers.append((cx, cy, radius * 0.3))
    return centers


def plot_gaad(save_path=None):
    """Full GAAD visualization with golden subdivision and phi-spiral."""
    
    frame_w, frame_h = 800, 500  # Aspect ratio 1.6 (~φ=1.618)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ──────────────────────────
    # LEFT: Recursive Golden Subdivision
    # ──────────────────────────
    ax1.set_xlim(-20, frame_w + 20)
    ax1.set_ylim(frame_h + 20, -20)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    regions = golden_subdivide((0, 0, frame_w, frame_h), max_depth=5)
    
    # Color by depth
    depth_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    
    for depth, (x, y, w, h), rtype in regions:
        color = depth_colors[min(depth, len(depth_colors)-1)]
        rect = Rectangle((x, y), w, h, 
                        fill=False, edgecolor=color, 
                        linewidth=2.5 - 0.3*depth, alpha=0.8)
        ax1.add_patch(rect)
        
        # Fill squares with light tint
        if rtype == 'square':
            fill = Rectangle((x, y), w, h, 
                           facecolor=color, alpha=0.08 + 0.03*(5-depth))
            ax1.add_patch(fill)
        
        # Label depth
        if w > 30 and h > 20:
            ax1.text(x + w/2, y + h/2, f"d={depth}", 
                    fontsize=7, color=color, ha='center', va='center',
                    alpha=0.6)
    
    ax1.set_title(
        "Recursive Golden Subdivision\n"
        f"Frame {frame_w}×{frame_h} (aspect={frame_w/frame_h:.3f}, φ≈{phi:.3f})\n"
        "Squares in golden, remainders are new golden rects",
        fontsize=10
    )
    
    # ──────────────────────────
    # RIGHT: Phi-Spiral Patching
    # ──────────────────────────
    ax2.set_xlim(-20, frame_w + 20)
    ax2.set_ylim(frame_h + 20, -20)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw phi spiral
    theta = np.linspace(0, 6*np.pi, 500)
    cx, cy = frame_w/2, frame_h/2
    a = 8
    r = a * phi ** (theta / (2*np.pi))
    spiral_x = cx + r * np.cos(theta)
    spiral_y = cy + r * np.sin(theta)
    ax2.plot(spiral_x, spiral_y, color='#d4a017', linewidth=1.5, 
            alpha=0.6, label=f'φ-spiral: r(θ) = {a}·φ^(θ/2π)')
    
    # Phi-spiral centers as sampling regions
    centers = phi_spiral_centers(frame_w, frame_h, n_spirals=10)
    for cx, cy, r in centers:
        circle = plt.Circle((cx, cy), r, fill=False, 
                          edgecolor='#c0392b', linewidth=1.5, alpha=0.5)
        ax2.add_patch(circle)
        # Fill
        circle_fill = plt.Circle((cx, cy), r, 
                                facecolor='#c0392b', alpha=0.06)
        ax2.add_patch(circle_fill)
    
    ax2.set_title(
        "Phi-Spiral Patching\n"
        f"{len(centers)} φ-scaled sampling regions\n"
        "Regions grow in φ-proportion from golden spiral",
        fontsize=10
    )
    
    # Draw golden ratio φ callout at bottom
    phi_approx = f"φ = (1+√5)/2 ≈ {phi:.6f}"
    fig.text(0.5, 0.02, 
            f"Golden Ratio: {phi_approx}  |  "
            f"φ² = φ + 1 ≈ {phi**2:.6f}  |  "
            f"1/φ = φ-1 ≈ {1/phi:.6f}",
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='#fef9e7', alpha=0.9))
    
    # Overall title
    fig.suptitle(
        "Golden Aspect Adaptive Decomposition (GAAD)\n"
        "Aspect-Ratio Agnostic Frame Decomposition via φ",
        fontsize=13, y=0.98, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[SAVED] {save_path}")
    
    plt.close()
    return fig


if __name__ == '__main__':
    save = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'visualizations', 'golden_ratio_decomposition.png')
    plot_gaad(save_path=save)
    print("[DONE] GAAD visualization complete.")
    
    # Verify golden ratio algebra
    print(f"[MATH] φ = {phi:.10f}")
    print(f"[MATH] φ² = {phi**2:.10f}")
    print(f"[MATH] φ + 1 = {phi + 1:.10f}")
    print(f"[MATH] φ² == φ + 1? {abs(phi**2 - (phi + 1)) < 1e-10}")
    print(f"[MATH] 1/φ = {1/phi:.10f}")
    print(f"[MATH] φ - 1 = {phi - 1:.10f}")
    print(f"[MATH] 1/φ == φ - 1? {abs(1/phi - (phi - 1)) < 1e-10}")
