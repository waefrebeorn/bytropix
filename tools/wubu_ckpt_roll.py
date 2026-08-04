#!/usr/bin/env python3
"""wubu_ckpt_roll.py — agnostic rolling-checkpoint retention (the file-bloat fix).

Checkpoint directories grow forever because nothing rotates them. This tool
keeps only the NEWEST N step-checkpoints per checkpoint LINE (default 3),
tightens to 2 when the directory gets large (default > 4 GB), and deletes —
or cold-archives with --move-to — the rest. It is AGNOSTIC: it works on ANY
checkpoint directory and naming scheme; the LINE is derived from the
filename, never hardcoded (works for wubuwizard seed-*.st lines, the
seed-sft.st-* SFT line, wubuos bear_*.ckpt, anything).

Line = filename with the `-NNNN.st` step suffix stripped. Files with NO step
suffix are ANCHORS (a generation's final/seed state) and are always kept.

Usage:
  python3 tools/wubu_ckpt_roll.py <checkpoint-dir> [options]

Options:
  --keep N       step checkpoints kept per line when not large (default 3)
  --tighten N    step checkpoints kept per line when large (default 2)
  --max-gb X     total dir size (GiB) that triggers tightening (default 4.0)
  --move-to DIR  MOVE pruned files here instead of deleting (cold archive;
                 e.g. the SD card preserves the DGM lineage ledger)
  --pattern GLOB only consider files matching this glob (default '*.st')
  --dry-run      print the plan, change nothing (exit 0)
  --verbose      print every file acted on
"""

import argparse
import glob
import os
import re
import shutil
import sys

STEP_RE = re.compile(r"^(?P<line>.+)-(?P<step>\d{3,6})\.st$")


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Rolling checkpoint retention: keep the newest N step-"
                    "checkpoints per line; tighten to 2 when the dir is large.")
    ap.add_argument("dir", help="checkpoint directory to clean")
    ap.add_argument("--keep", type=int, default=3, help="keep per line (default 3)")
    ap.add_argument("--tighten", type=int, default=2, help="keep per line when large (default 2)")
    ap.add_argument("--max-gb", type=float, default=4.0,
                    help="total dir GiB that triggers tightening (default 4.0)")
    ap.add_argument("--move-to", default=None,
                    help="move pruned files here instead of deleting (cold archive)")
    ap.add_argument("--pattern", default="*.st", help="file glob (default '*.st')")
    ap.add_argument("--dry-run", action="store_true", help="plan only, change nothing")
    ap.add_argument("--verbose", action="store_true", help="print every file acted on")
    return ap.parse_args(argv)


def prune_one(path, args, kept_anyway):
    if args.dry_run:
        print("  [dry] prune %s%s" % (path, "  (kept anyway)" if kept_anyway else ""))
        return
    if args.move_to:
        os.makedirs(args.move_to, exist_ok=True)
        dst = os.path.join(args.move_to, os.path.basename(path))
        try:
            shutil.move(path, dst)
            if args.verbose:
                print("  move  %s -> %s" % (path, dst))
        except OSError as e:
            print("  ERROR moving %s: %s" % (path, e), file=sys.stderr)
    else:
        try:
            os.remove(path)
            if args.verbose:
                print("  del   %s" % path)
        except OSError as e:
            print("  ERROR deleting %s: %s" % (path, e), file=sys.stderr)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not os.path.isdir(args.dir):
        print("error: %s is not a directory" % args.dir, file=sys.stderr)
        return 2
    if args.keep < 1 or args.tighten < 1:
        print("error: --keep/--tighten must be >= 1", file=sys.stderr)
        return 2

    files = [f for f in glob.glob(os.path.join(args.dir, args.pattern))
             if os.path.isfile(f)]
    if not files:
        print("%s: no files match '%s'" % (args.dir, args.pattern))
        return 0

    lines, anchors = {}, []
    for f in files:
        m = STEP_RE.match(os.path.basename(f))
        if m:
            lines.setdefault(m.group("line"), []).append((int(m.group("step")), f))
        else:
            anchors.append(f)

    total = sum(os.path.getsize(f) for f in files)
    keep = args.keep if total <= args.max_gb * (1024 ** 3) else args.tighten
    state = "normal" if keep == args.keep else "LARGE (%.1f GiB > %.1f)" % (
        total / (1024 ** 3), args.max_gb)

    print("%s: %d files, %.2f GiB total, %d line(s), %d anchor(s) — keep %d/line [%s]"
          % (args.dir, len(files), total / (1024 ** 3), len(lines), len(anchors),
             keep, state))
    for f in anchors:
        if args.verbose:
            print("  keep %s (anchor, always kept)" % f)

    pruned = 0
    for line, cks in sorted(lines.items()):
        cks.sort(key=lambda t: t[0])
        n_keep = min(keep, len(cks))
        for step, f in cks[:-n_keep]:
            prune_one(f, args, kept_anyway=False)
            pruned += 1
        if args.verbose:
            for step, f in cks[-n_keep:]:
                print("  keep %s (step %d)" % (f, step))

    print("pruned %d file(s) (%s)." % (pruned,
          "dry run — nothing changed" if args.dry_run else
          ("archived to %s" % args.move_to if args.move_to else "deleted")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
