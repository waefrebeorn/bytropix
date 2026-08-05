#!/usr/bin/env python3
"""
Generate compile_commands.json for clangd / IDE / agent tooling.
Scans the source tree for .c and .cpp files and emits one entry each
with the standard CFLAGS. This is a best-effort generator that mirrors
what `bear -- make` would produce but needs no extra dependencies.

Usage: python3 tools/gen_compile_commands.py
"""
import os, json, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFLAGS = "-O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -I/usr/include -fopenmp"

entries = []
for dirpath, _, filenames in os.walk(os.path.join(ROOT, 'src')):
    for fname in filenames:
        if fname.endswith('.c'):
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, ROOT)
            obj = rel.replace('.c', '.o')
            entries.append({
                "directory": ROOT,
                "command": f"gcc {CFLAGS} -c -o {obj} {rel}",
                "file": rel,
                "output": obj
            })
for dirpath, _, filenames in os.walk(os.path.join(ROOT, 'tools')):
    for fname in filenames:
        if fname.endswith('.cpp'):
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, ROOT)
            base = os.path.splitext(rel)[0]
            entries.append({
                "directory": ROOT,
                "command": f"g++ {CFLAGS} -std=c++17 -c -o {base}.o {rel}",
                "file": rel,
                "output": f"{base}.o"
            })

out = os.path.join(ROOT, 'compile_commands.json')
with open(out, 'w') as f:
    json.dump(entries, f, indent=2)
print(f"Wrote {len(entries)} entries to {out}")
