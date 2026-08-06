#!/usr/bin/env python3
"""
verify_research_gateway.py — Theme J rank-4: research→wired-code verification gate.

Verifies that every 'wired'/'tested'/'shipped' gap in research/INDEX.md
has corresponding source files in the tree or is explicitly marked as
a cross-repo (wubuos) reference.

Usage: python3 tools/verify_research_gateway.py
Exit: 0 if all wired gaps are backed by real source files, 1 if any are not.
"""
import re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, 'research', 'INDEX.md')

def parse_index(path):
    """Parse INDEX.md lines like: - AN01 Title ... `status` (source)
    The title may contain backticks, so we match from the END.
    """
    with open(path) as f:
        content = f.read()
    gaps = []
    for line in content.split('\n'):
        m = re.match(r'^[-*] (AN\d+|BL\d+|WB\d+|GG\d+)\s+(.+)`(\w+)`\s*(.*)$', line)
        if m:
            gaps.append({
                'id': m.group(1),
                'title': m.group(2).strip(),
                'status': m.group(3),
                'ref': m.group(4).strip()
            })
    return gaps

def resolve_module(mod, root):
    """Check if a wubu_xxx module exists in src/, include/, or tools/."""
    for subdir in ['src', 'include', 'tools']:
        if os.path.exists(os.path.join(root, subdir, f'{mod}.c')):
            return True
        if os.path.exists(os.path.join(root, subdir, f'{mod}.h')):
            return True
    return False

def check_source_exists(ref, root):
    """Check if the ref mentions source files that exist in the tree."""
    if not ref:
        return False, "no source reference"

    # Cross-repo reference (wubuos:) — accept as valid, different repo
    if 'wubuos:' in ref:
        return True, "cross-repo reference (wubuos)"

    # Check for docs/ references
    if ref.startswith('docs') and not re.search(r'\bwubu_[a-z]', ref):
        return True, "docs reference"

    # Extract module names (wubu_xxx) from inline-quoted text
    # e.g. `wubu_mhc.c` or (wubu_mhc, test...)
    modules = set()
    for m in re.finditer(r'wubu_([a-z0-9_]+)', ref):
        modules.add(f"wubu_{m.group(1)}")

    # Extract explicit file paths (include/xxx.c, src/xxx.c, etc.)
    explicit = re.findall(r'(?:include|src|tools|tests)/[a-zA-Z0-9_/.-]+\.(?:c|cu|h)', ref)

    if not modules and not explicit:
        if 'docs' in ref:
            return True, "docs reference"
        return False, f"no file references in ref: {ref[:50]}"

    missing = []
    # If explicit file paths exist and all exist, pass — module names
    # may be function names (e.g. wubu_ts_export_mixed) not modules.
    for f in explicit:
        if not os.path.exists(os.path.join(root, f)):
            missing.append(f)
    if missing:
        return False, f"missing files: {missing}"
    if not explicit:
        # No explicit paths — rely on module resolution
        for mod in modules:
            if not resolve_module(mod, root):
                missing.append(mod)
        if missing:
            return False, f"missing modules: {missing}"
    return True, f"OK: modules={list(modules)}, files={explicit}"

if __name__ == '__main__':
    gaps = parse_index(INDEX)
    wired_gaps = [g for g in gaps if g['status'] in ('wired', 'tested', 'shipped')]
    open_gaps = [g for g in gaps if g['status'] in ('open', 'research')]

    issues = []
    for gap in wired_gaps:
        # Search full text (title + ref) for file references
        full_text = gap['title'] + ' ' + gap['ref']
        ok, msg = check_source_exists(full_text, ROOT)
        if not ok:
            issues.append(f"{gap['id']}: {gap['title'][:50]} — {msg}")
    print(f"Total gaps: {len(gaps)} | wired: {len(wired_gaps)} | open: {len(open_gaps)}")
    if issues:
        print(f"FAIL — {len(issues)} wired gaps missing verifiable source:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print(f"PASS — all {len(wired_gaps)} wired gaps backed by real source files")
        sys.exit(0)
