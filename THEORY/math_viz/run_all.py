#!/usr/bin/env python3
"""
math_viz/run_all.py — Regenerate all math diagrams from first principles.

Usage:
    python3 math_viz/run_all.py

Output: visualizations/*.png
"""

import subprocess, sys, os

SCRIPTS = [
    "01_nested_hyperbolic_spaces.py",
    "02_golden_ratio_decomposition.py",
    "03_poincare_clock.py",
    "04_lie_group_nesting.py",
]

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(os.path.dirname(BASE), "visualizations")
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 60)
print("Bytropix Math Visualization Generator")
print("=" * 60)

for script in SCRIPTS:
    path = os.path.join(BASE, script)
    print(f"\n--- Running {script} ---")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True,
        timeout=120
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed: {result.stderr[:200]}")
    else:
        print(f"[OK] {script} completed successfully")

print("\n" + "=" * 60)
print("All diagrams generated in visualizations/")
print("=" * 60)
print("\nFiles:")
for f in sorted(os.listdir(OUTPUT)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(OUTPUT, f))
        print(f"  {f:45s} {size/1024:.0f} KB")
