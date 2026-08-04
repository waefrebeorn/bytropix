#!/usr/bin/env python3
"""repodoc — agnostic repo documentation + hygiene generator.

One command to audit a repo, refresh its README/TOPOLOGY/STATUS docs, and
report push state — works on ANY repo (wubuwizard, wubuos, ...). Stdlib only.

Usage:
  repodoc.py <repo-path> [--audit|--map|--readme|--topology|--all] [--push "msg"]

Subcommands:
  --audit      git state + artifact scan (ELF/big/venv) + .gitignore coverage
               + README make-target claims vs Makefile reality
  --map        module map: src/*.c one-liners (header comment), include/,
               tools/test_*.c, research/*.md + INDEX
  --readme     regenerate README.md from the map (template + repo config)
  --topology   regenerate docs/TOPOLOGY.md module sections
  --status     append verified-claims template to STATUS.md (from make test)
  --all        audit + map + readme + topology (default)
  --push "msg" git add -A + commit + push (run AFTER regenerating docs)

Per-repo config: <repo>/.repodoc.json — {name, tagline, desc, extras:{...}}
Everything writes only the files it owns (README.md, docs/TOPOLOGY.md) with
clear BEGIN/END markers so hand-written prose survives regeneration.
"""
import json, os, re, subprocess, sys, datetime

BEGIN = "<!-- repodoc:BEGIN -->"
END = "<!-- repodoc:END -->"

def sh(cmd, cwd, text=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

def git(cwd, *args):
    rc, out, err = sh(["git"] + list(args), cwd)
    return out.strip()

def audit(repo):
    print(f"## AUDIT {repo}")
    rc, out, _ = sh(["git", "status", "-sb"], repo)
    print(out.strip() or "(clean)")
    # branch tracking / push state
    br = git(repo, "branch", "--show-current")
    upstream = git(repo, "rev-parse", f"--abbrev-ref", f"{br}@{{u}}")
    if "fatal" in upstream:
        print(f"  ! branch '{br}' has NO upstream — push with: git push -u origin {br}")
    else:
        ahead = git(repo, "rev-list", "--count", f"@{'{upstream}'}" if False else f"{br}..{upstream}")
        # simpler: compare local vs origin
        rc, _, _ = sh(["git", "diff", "--quiet", f"origin/{br}..{br}"], repo)
        print(f"  upstream: {upstream}  {'AHEAD of origin (unpushed)' if rc != 0 else 'in sync with origin'}")
    # tracked artifacts
    tracked = git(repo, "ls-files").splitlines()
    els, big, venv, misc = [], [], [], []
    for f in tracked:
        if not os.path.exists(f): continue
        if os.path.isfile(f):
            try:
                with open(f, "rb") as fh: head = fh.read(4)
                if head == b"\x7fELF": els.append(f)
            except Exception: pass
            sz = os.path.getsize(f)
            if sz > 5_000_000: big.append((f, sz))
        if f.startswith(("venv", ".venv", ".local/", "dist/", "__pycache__/", ".hermes/")):
            venv.append(f)
        if f.endswith((".o", ".zst", ".bin", ".mp4", ".deb", ".iso", ".pyc")):
            misc.append(f)
    print(f"  tracked: {len(tracked)} files | ELF bins: {len(els)} | >5MB: {len(big)} | venv/profile: {len(venv)} | misc: {len(misc)}")
    for f in els[:8]: print(f"    ELF: {f}")
    for f, sz in big[:8]: print(f"    BIG {sz/1e6:7.1f}MB: {f}")
    # README make-claims vs Makefile
    mf = os.path.join(repo, "Makefile")
    readme = os.path.join(repo, "README.md")
    if os.path.exists(mf) and os.path.exists(readme):
        mk = open(mf).read()
        claims = set(re.findall(r"make\s+([\w_\-/]+)", open(readme).read()))
        claims -= {"targets", "it", "sure", "the", "a", "an", "docs", "all", "test", "test_", "hosted"}  # prose noise
        missing = sorted(c for c in claims if not re.search(rf"^{re.escape(c)}\s*:", mk, re.M))
        print(f"  README make-claims: {len(claims)} | missing from Makefile: {missing or 'none'}")
    return {"tracked": len(tracked), "elf": len(els)}

def _desc(txt, fname):
    """First clean prose line from the header comment block."""
    # candidate lines from the leading comment block
    cands = []
    for m in re.finditer(r"^\s*(?:\*|\*/|//|/\*)?\s*(.{20,140})$", txt, re.M):
        l = m.group(1).strip()
        if not l: continue
        if l.startswith(("==", "--", "**", "* ", "//", "/*", "*/", "#")): continue
        if l == fname or l.startswith(fname): continue
        if re.match(r"^[\s=*#/\-]+$", l): continue
        if "repodoc" in l.lower(): continue
        cands.append(l)
        if len(cands) >= 3: break
    for c in cands:
        if len(c) > 30 and not c.startswith(("The ", "A ", "An ")) or len(c) > 60:
            return c[:110]
    return cands[0][:110] if cands else ""

def module_map(repo):
    """src/**/*.c (recursive, depth 2) + include/*.h + tools/test_*.c + research."""
    out = {"src": [], "tests": [], "research": []}
    srcdir = os.path.join(repo, "src")
    if os.path.isdir(srcdir):
        for root, dirs, files in os.walk(srcdir):
            depth = root[len(srcdir):].count(os.sep)
            if depth > 1: continue
            for f in sorted(files):
                if f.endswith(".c") and not f.endswith(".o"):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, repo)
                    txt = open(p, errors="replace").read()
                    out["src"].append((rel, _desc(txt, f)))
    tdir = os.path.join(repo, "tools")
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if f.startswith("test_") and f.endswith(".c"):
                out["tests"].append(f[:-2])
    rdir = os.path.join(repo, "research")
    if os.path.isdir(rdir):
        for f in sorted(os.listdir(rdir)):
            if re.match(r"\d{3}-", f) and f.endswith(".md"):
                txt = open(os.path.join(rdir, f), errors="replace").read()
                m = re.search(r"^#\s+(.+)", txt, re.M)
                title = m.group(1).strip() if m else f
                out["research"].append((f, title))
    return out

def map_rows(repo, m):
    rows = []
    for f, desc in m["src"]:
        rows.append(f"| `src/{f}` | {desc[:100]} |")
    return rows

def readme_sections(repo, m, cfg):
    """Compact module index for the README; the FULL table lives in docs/MODULES.md.
    (DA 2026-08-04: bolting 300+ rows onto the README drowned the hand-written
    prose — keep README tight, full table in MODULES.md.)"""
    mk = open(os.path.join(repo, "Makefile")).read()
    targets = sorted(set(re.findall(r"^([\w_\-/]+)\s*:", mk, re.M)))
    tests = [t for t in targets if t.startswith("test_")]
    rrows = "\n".join(f"| [{f}](research/{f}) | {t[:90]} |" for f, t in m["research"])
    sec = f"""
## Module index (auto-generated {datetime.date.today()})

- **{len(m['src'])} C modules** — full annotated table: [docs/MODULES.md](docs/MODULES.md)
- **{len(m['tests'])} test tools** (make targets `test_*`, e.g. `{', '.join(tests[:10])}...`)
- **{len(m['research'])} research docs** — full ledger: [research/INDEX.md](research/INDEX.md)

Regenerate with: `python3 tools/repodoc/repodoc.py . --readme`
"""
    return sec

def modules_md(repo, m):
    rows = "\n".join(f"| `{f}` | {d[:90]} |" for f, d in m["src"])
    return f"""# Module Map (auto-generated {datetime.date.today()})

Full annotated table of `src/` modules. Regenerate with
`python3 tools/repodoc/repodoc.py . --modules`.

| File | Purpose |
|---|---|
{rows}
"""

def write_section(path, section):
    """Insert/replace the repodoc block in a markdown file, preserving prose."""
    if os.path.exists(path):
        txt = open(path).read()
        if BEGIN in txt:
            txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), BEGIN + "\n" + section.strip() + "\n" + END, txt, flags=re.S)
        else:
            txt = txt.rstrip() + "\n\n" + BEGIN + "\n" + section.strip() + "\n" + END + "\n"
    else:
        txt = "# " + os.path.basename(os.path.dirname(path)) + "\n\n" + BEGIN + "\n" + section.strip() + "\n" + END + "\n"
    open(path, "w").write(txt)
    print(f"  wrote {path}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return 1
    repo = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]
    push_msg = None
    if "--push" in args:
        i = args.index("--push"); push_msg = args[i+1]; args = args[:i] + args[i+2:]
    mode = "all" if not any(a.startswith("--") and a != "--all" for a in args) else "all"
    # honor explicit flags
    want = set(a.lstrip("-") for a in args if a.startswith("--"))
    cfg = {}
    if os.path.exists(os.path.join(repo, ".repodoc.json")):
        cfg = json.load(open(os.path.join(repo, ".repodoc.json")))
    do_audit = "audit" in want or "all" in want or not want
    do_map = "map" in want or "all" in want
    if do_audit: audit(repo)
    m = module_map(repo) if (do_map or "readme" in want or "topology" in want or "all" in want) else {}
    if "readme" in want or "all" in want:
        sec = readme_sections(repo, m, cfg)
        write_section(os.path.join(repo, "README.md"), sec)
    if "modules" in want or "all" in want:
        os.makedirs(os.path.join(repo, "docs"), exist_ok=True)
        write_section(os.path.join(repo, "docs", "MODULES.md"), modules_md(repo, m))
    # TOPOLOGY.md is a CURATED document — repodoc never rewrites it (DA 2026-08-04:
    # bolting a machine table onto the curated map destroyed its signal).
    if push_msg:
        subprocess.run(["git", "add", "-A"], cwd=repo)
        r = subprocess.run(["git", "commit", "-q", "-m", push_msg], cwd=repo, capture_output=True, text=True)
        print("commit:", "ok" if r.returncode == 0 else r.stderr[:200])
        r2 = subprocess.run(["git", "push"], cwd=repo, capture_output=True, text=True)
        print("push:", "ok" if r2.returncode == 0 else r2.stderr[-200:])
    print("repodoc done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
