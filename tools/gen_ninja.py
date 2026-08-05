#!/usr/bin/env python3
"""Generate build.ninja from compile_commands.json (research 066-C5)."""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cc_path = os.path.join(ROOT, 'compile_commands.json')
cc = json.load(open(cc_path)) if os.path.exists(cc_path) else []

lines = [
    "ninja_required_version = 1.5",
    "cc = gcc",
    "cxx = g++",
    'cflags = -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -fopenmp',
    "",
]
for e in cc:
    obj = e.get('output', '')
    src = e.get('file', '')
    cmd = e.get('command', '')
    if obj and src:
        lines.append(f"build {obj}: cc {src}")
        lines.append(f"  command = {cmd}")
        lines.append("")

with open(os.path.join(ROOT, 'build.ninja'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"Wrote build.ninja with {len(cc)} edges")
