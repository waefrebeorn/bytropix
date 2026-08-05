#!/usr/bin/env python3
"""
Generate docs/symbols.json — a symbol index of all function definitions,
typeedefs, and structs across the wubuwizard source tree.
Usage: python3 tools/gen_symbols.py > docs/symbols.json
"""
import os, sys, re, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
entries = []

for dirpath, _, filenames in os.walk(os.path.join(ROOT, 'src')):
    for fname in filenames:
        if not (fname.endswith('.c') or fname.endswith('.h')):
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            content = open(fpath, errors='replace').read()
        except:
            continue
        rel = os.path.relpath(fpath, ROOT)
        for m in re.finditer(r'^(\w[\w\s\*]+)\s+(\w+)\s*\([^;{]*\)\s*\{', content, re.M):
            entries.append({
                "kind": "function_def",
                "name": m.group(2),
                "signature": m.group(0).strip(),
                "file": rel,
                "line": content[:m.start()].count('\n') + 1
            })
        for m in re.finditer(r'typedef\s+(struct|enum)\s+(\w+)\s+(\w+);', content):
            entries.append({"kind": "typedef", "name": m.group(3), "target": f"{m.group(1)} {m.group(2)}", "file": rel})
        for m in re.finditer(r'typedef\s+struct\s+(\w+)\s*\{', content):
            entries.append({"kind": "tagged_struct", "name": m.group(1), "file": rel})

json.dump(entries, sys.stdout, indent=2)
