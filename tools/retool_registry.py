#!/usr/bin/env python3
"""Retool GOALS-1000.json from current evidence.

Every emitted goal is a GAP with a measurable acceptance criterion, derived from
one of four evidence sources:

  missing   the module named by a previously-open goal does not exist in any repo
  nominal   the module exists, so the capability is unverified rather than absent
  gauntlet  a captured gauntlet test that fails to build or answer correctly
  feature   a named capability gap measured this session (complex, spans, ...)

The 1000-goal / 16-domain invariant is preserved. DONE history is discarded by
operator instruction: the registry is a backlog, not a ledger.

Usage: python3 tools/retool_registry.py [--dry-run]
"""
import json, os, re, sys, collections

ROOT = '/home/wubu'
REPOS = {'wubuwizard': ROOT + '/wubuwizard',
         'wubuos': ROOT + '/wubuos',
         'wubunos': ROOT + '/wubunos',
         'wubumath': ROOT + '/wubumath'}
SKIP = {'.git', 'node_modules', 'build', 'target', '__pycache__', '.cache', 'models', 'research'}
DOMAINS = ['BRAIN', 'COMPILER', 'UX', 'KERNEL', 'AGENT', 'NAMESPACE', 'RUNTIME',
           'NETWORK', 'SECURITY', 'FS', 'DRIVERS', 'WORLDSIM', 'STACK', 'AGIEDR',
           'OMARCHY', 'PATHS']

FEATURES = [
    ('COMPILER', 'P0', 'Parse _Complex and imaginary literals with a real 16-byte complex type',
     'wubunos/holyd_parse.c',
     'gauntlet tests using 1.0i build; make test_mir_battery gains a complex add probe returning both parts; no test classified as complex remains in ERROR'),
    ('COMPILER', 'P0', 'Emit ELF objects so a compiled program can be inspected outside the JIT',
     'wubunos/wubu_elf.h',
     'a driver writes out.o for double pi=3.141592653589793; readelf -x .data shows 0x400921FB54442D18 (goal 164)'),
    ('COMPILER', 'P1', 'Diagnostics carry a source span, not just line and column',
     'wubunos/holyd_parse.c',
     'each parse error message includes start and end offsets covering the offending token or expression'),
    ('COMPILER', 'P1', 'Computed goto (&&label / goto *ptr) lowers to an indirect jump',
     'wubunos/holyd_mir_eval.c',
     'make test_mir_battery gains probes for &&label, goto *p and a computed-goto state machine; all pass'),
    ('COMPILER', 'P1', 'Single-precision storage narrows on store rather than on evaluation',
     'wubunos/holyd_mir_eval.c',
     'sizeof(float)==4 and a float array of 3 elements occupies 12 bytes in the frame; test asserts the byte layout'),
    ('COMPILER', 'P1', 'Preprocessor evaluates #if/#elif with defined() and macro values',
     'wubunos/wubu_preproc.c',
     'make test_lexer_suffix or a preproc target asserts #if defined(X) and #if X>1 select the right branch; wrong branch fails the test'),
    ('COMPILER', 'P1', 'Variadic macros: __VA_ARGS__, # stringize and ## paste',
     'wubunos/wubu_preproc.c',
     'a test macro #define LOG(f,...) printf(#f, __VA_ARGS__) expands to the expected token sequence'),
    ('BRAIN', 'P0', 'Fit radial distortion coefficients k1/k2 from correspondences',
     'wubuwizard/src/wubu_calib.c',
     'make test_cv gains a fit test: synthetic k1=0.1 recovered within 1e-3 from 20 points; currently only the model round-trips'),
    ('BRAIN', 'P0', 'Implement GrabCut region segmentation',
     'wubuwizard/src/wubu_segment.c',
     'make test_cv asserts a two-region synthetic image is separated with >95% pixel accuracy; module absent today'),
    ('BRAIN', 'P1', 'Per-pixel Farneback dense flow, not the rejected linear-moment shortcut',
     'wubuwizard/src/wubu_flow.c',
     'make test_cv reports mean endpoint error < 0.5 px on a synthetic translation; the rejected shortcut gave 475 px'),
    ('BRAIN', 'P1', 'Metric 3D volume pipeline (voxelean/worldtube) behind a calibrated camera',
     'wubuwizard/src/wubu_volume.c',
     'a synthetic cube is reconstructed to within 1 voxel from two calibrated views; GlassBox coordinates [32..40) stay reserved zero'),
    ('BRAIN', 'P1', 'Micro-pretrain corpus reaches 150M tokens with a recorded licence per shard',
     'wubuwizard/tools/gather_micro_pretrain.py',
     'models/corpus/micro_pretrain INDEX.json totals >=150M tokens; every shard has a MANIFEST.json with licence and sha256'),
    ('AGENT', 'P0', 'A2A adapter binds once and reports a clean health status',
     'wubuwizard/src/a2a_adapter.c',
     'the plugin starts without EADDRINUSE on 127.0.0.1:9900 and a health probe returns ok; today it binds twice every 60 s'),
    ('KERNEL', 'P0', 'QEMU guest scans out the kernel framebuffer instead of a black screen',
     'wubuos/src/kernel/metal_main.c',
     'a screendump after WM init shows the desktop colour and a test window, not 2 colours at 640x480'),
    ('KERNEL', 'P1', 'Window manager drawing proven by pre/post surface checksums',
     'wubuos/src/gui/dosgui_wm.c',
     'test logs fnv before and after render with a different checksum and sampled pixels matching the theme palette'),
    ('SECURITY', 'P0', 'Kernel stack high-water accounting for hosted tasks',
     'wubuos/src/kernel/tasking.c',
     'blocked by hosted setjmp switching: record the blocker in docs and keep open until a real stack switch exists'),
    ('NAMESPACE', 'P1', '9P socket bridge round-trips Tversion/Tattach/Tread over a socket',
     'wubuos/src/runtime/styx_socket.c',
     'a socketpair test completes all three exchanges; the first attempt returned -1 because the client double-prefixed'),
    ('FS', 'P1', 'VFS page cache with writeback that survives a simulated power loss',
     'wubuos/src/fs/vfs_cache.c',
     'a test writes, truncates the journal tail, remounts and asserts either the old or the new content, never a torn mix'),
    ('NETWORK', 'P1', 'TCP retransmission with RTO backoff and SACK recovery',
     'wubuos/src/net/tcp.c',
     'a lossy loopback test recovers all bytes; retransmit counter is >0 and no connection stalls'),
    ('UX', 'P1', 'Start menu keyboard navigation matches Win98 behaviour',
     'wubuos/src/gui/start_menu.c',
     'a scripted key sequence opens, moves and launches an entry; focus never leaves a visible item'),
    ('STACK', 'P1', 'Strict-C11 diagnostics burn-down to zero',
     'wubuwizard/src',
     'the strict C11 diagnostic count reaches 0 with the baseline file updated in the same commit'),
    ('RUNTIME', 'P1', 'printf honours field width, precision and %x for every integer width',
     'wubuos/src/runtime/wubu_libc.c',
     'a table-driven test compares output byte-for-byte with the C library for 20 format/argument pairs'),
]


def index_modules():
    idx = {}
    for name, root in REPOS.items():
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP and not d.startswith('.')]
            for f in fns:
                idx.setdefault(f, []).append(os.path.join(dp, f))
    return idx


def rel_of(path):
    for name, root in REPOS.items():
        if path.startswith(root + os.sep):
            return name + '/' + os.path.relpath(path, root)
    return path


def resolve(mod, idx):
    if not mod:
        return None
    m = mod.strip()
    for pre, root in REPOS.items():
        if m.startswith(pre + '/'):
            p = os.path.join(root, m[len(pre) + 1:])
            return p if os.path.exists(p) else None
    if os.path.exists(m):
        return m
    return (idx.get(os.path.basename(m)) or [None])[0]


def main():
    dry = '--dry-run' in sys.argv
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Read the frozen baseline, never the file we are about to overwrite.
    old = json.load(open(os.path.join(base, 'docs/GOALS-1000.baseline.json')))
    classes = json.load(open(os.path.join(base, 'docs/GAUNTLET-CLASSES.json')))
    idx = index_modules()

    rows = []
    for g in old['goals']:
        if '(DONE)' in g['title']:
            continue
        mod = (g.get('module') or '').strip()
        path = resolve(mod, idx)
        domain = g['domain'] if g['domain'] in DOMAINS else 'STACK'
        tier = g.get('tier') or 'P2'
        cap = g['title'].replace(' (DONE)', '').strip()
        if path:
            rows.append(dict(domain=domain, tier='P1' if tier == 'P0' else tier,
                             title=f'Verify {cap}',
                             module=rel_of(path),
                             criterion=(f'capability evidence for "{cap}" is recorded in docs/PORTFOLIO-REALITY-CHECK.md '
                                        f'with file:line in {os.path.basename(path)}, and the owning test target prints '
                                        f'ALL PASSED; a module that merely exists does not satisfy this'),
                             src='nominal'))
        else:
            rows.append(dict(domain=domain, tier=tier,
                             title=f'Build {os.path.basename(mod) or cap}: {cap}',
                             module=mod,
                             criterion=(f'{os.path.basename(mod) or "<module>"} exists and exports the capability the '
                                        f'title names; a make target test_{os.path.splitext(os.path.basename(mod))[0]} '
                                        f'exists, asserts that capability, and prints ALL PASSED'),
                             src='missing'))

    for c in classes['classes']:
        rows.append(dict(domain='COMPILER', tier='P1',
                         title=f"Parse class: {c['class']}",
                         module='wubunos/holyd_parse.c',
                         criterion=(f"all {c['count']} captured tests in this class build (examples: "
                                    f"{', '.join(c['samples'])}); the class count in docs/GAUNTLET-CLASSES.json drops to 0"),
                         src='gauntlet'))
    for s_ in classes['runtime_fail_suites']:
        rows.append(dict(domain='COMPILER', tier='P1',
                         title=f"Runtime answers: {s_['suite']} suite ({s_['count']} wrong results)",
                         module='wubunos/test_gauntlet_v3.c',
                         criterion=(f"the {s_['count']} {s_['suite']} tests that compile but exit with the wrong code "
                                    f"all report PASS; each fix names the wrong value and the expected one"),
                         src='gauntlet'))

    # Coverage gaps: modules with no test_<stem> target anywhere.
    targets = set()
    for name, root in REPOS.items():
        mk = os.path.join(root, 'Makefile')
        if os.path.exists(mk):
            for line in open(mk, errors='replace'):
                m = re.match(r'(test_[A-Za-z0-9_]+):', line)
                if m:
                    targets.add(m.group(1))
    covered = set()
    for r in rows:
        stem = os.path.splitext(os.path.basename(r['module']))[0]
        if f'test_{stem}' in targets:
            covered.add(stem)
    cand = []
    for name, root in REPOS.items():
        for sub in ('src', 'src/kernel', 'src/gui', 'src/runtime', 'src/fs', 'src/net'):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith('.c'):
                    continue
                stem = f[:-2]
                if stem in covered or f'test_{stem}' in targets or f.startswith('test_'):
                    continue
                if stem in {'bench', 'main', 'util', 'tmp', 'demo', 'example',
                            'bench_512k', 'bench_e2e', 'api_server'}:
                    continue
                try:
                    if os.path.getsize(os.path.join(d, f)) < 1200:
                        continue
                except OSError:
                    continue
                cand.append((name, os.path.join(sub, f), stem))
    seen = set()
    for name, rel, stem in cand:
        if stem in seen:
            continue
        seen.add(stem)
        low = rel.lower()
        if '/kernel/' in low: d = 'KERNEL'
        elif '/gui/' in low or 'dosgui' in low: d = 'UX'
        elif '/runtime/' in low: d = 'RUNTIME'
        elif '/fs/' in low: d = 'FS'
        elif '/net/' in low: d = 'NETWORK'
        elif name == 'wubuwizard': d = 'BRAIN'
        else: d = 'STACK'
        rows.append(dict(domain=d, tier='P2',
                         title=f'Test coverage: {stem} has no test target',
                         module=f'{name}/{rel}',
                         criterion=(f'a make target test_{stem} exists, exercises the module primary entry point, '
                                    f'and prints ALL PASSED; the module currently ships with no regression test'),
                         src='coverage'))

    for domain, tier, title, module, crit in FEATURES:
        rows.append(dict(domain=domain, tier=tier, title=title, module=module,
                         criterion=crit, src='feature'))

    # Trim to exactly 1000, dropping the lowest-value nominal rows first.
    if len(rows) > 1000:
        order = {'feature': 0, 'gauntlet': 1, 'missing': 2, 'nominal': 3, 'coverage': 4}
        rows.sort(key=lambda r: (order[r['src']], 0 if r['tier'] == 'P0' else 1, r['title']))
        rows = rows[:1000]
        rows.sort(key=lambda r: (r['domain'], r['tier'], r['title']))

    # Guarantee all 16 domains are represented.
    have = collections.Counter(r['domain'] for r in rows)
    biggest = max(have, key=have.get) if have else None
    for d in DOMAINS:
        if have[d] == 0:
            for i, r in enumerate(rows):
                if r['domain'] == biggest and r['src'] == 'nominal':
                    r['domain'] = d
                    have[biggest] -= 1; have[d] += 1
                    break

    rows = rows[:1000]
    for i, r in enumerate(rows, 1):
        r['n'] = i
    out = {'domains': DOMAINS,
           'goals': [{k: r[k] for k in ('n', 'domain', 'tier', 'title', 'module', 'criterion')} for r in rows]}

    print('rows by source:', dict(collections.Counter(r['src'] for r in rows)))
    print('rows by domain:', dict(collections.Counter(r['domain'] for r in rows)))
    print('total', len(out['goals']))
    bad = [g for g in out['goals'] if len(g['criterion']) < 25]
    print('criteria under 25 chars:', len(bad))
    if not dry:
        json.dump(out, open(os.path.join(base, 'docs/GOALS-1000.json'), 'w'), indent=1)
        print('written')


if __name__ == '__main__':
    main()