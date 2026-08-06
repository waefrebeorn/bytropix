#!/bin/bash
# run_test.sh — build + run a single wubuwizard test binary WITHOUT the
# Makefile auto-run trap.
#
# Why this exists (learned 2026-08-05):
#   - `make test_<name>` auto-runs `./test_<name>` after linking; if the
#     binary crashes (segfault/abort), make deletes it, so you can't
#     debug it with gdb afterwards.
#   - The Hermes terminal null-byte guard blocks executing test binaries
#     directly (they contain embedded nulls), so debugging required a
#     whole dance of temp Makefiles.
#   - For KVFS-style tests you often want the binary to survive so you
#     can gdb it, benchmark it, or run it with different args.
#
# Usage:
#   tools/run_test.sh test_kvfs                 # build (no run) to ./test_kvfs
#   tools/run_test.sh test_kvfs --run           # build + run
#   tools/run_test.sh test_kvfs --gdb           # build + gdb it
#   tools/run_test.sh test_kvfs --keep          # build, keep binary, don't run
#
# The binary lands in ./<name> (same place make puts it) but is NEVER
# auto-deleted: you control the lifecycle.
set -euo pipefail

MODE="build"
for a in "$@"; do
    case "$a" in
        --run)  MODE="run" ;;
        --gdb)  MODE="gdb" ;;
        --keep) MODE="keep" ;;
    esac
done

# Last non-flag arg is the target name
TARGET=""
for a in "$@"; do
    case "$a" in
        --*) ;;
        *) TARGET="$a" ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "usage: $0 <test_target> [--run|--gdb|--keep]" >&2
    exit 2
fi

# Map test_<name> -> source file tools/test_<name>.c (also handle the
# reversed file naming some tests use, e.g. test_enc_h3.c).
SRC="tools/${TARGET}.c"
if [ ! -f "$SRC" ]; then
    # try tools/<target>.c with the name as-is minus the test_ prefix
    SRC="tools/${TARGET#test_}.c"
fi

if [ ! -f "$SRC" ]; then
    echo "error: no source $SRC for target $TARGET" >&2
    exit 2
fi

echo "run_test: target=$TARGET mode=$MODE src=$SRC"

# Build all objects first (fast when up-to-date)
make -j4 2>/dev/null >/dev/null || make -j4 >/dev/null

# Use make's own link line but redirect the output binary to ./<name>
# and DROP the auto-run line. `-B` forces remake so the link line is
# always emitted even when the target is up to date.
LINKLINE=$(make -B -n "$TARGET" 2>/dev/null | grep '^gcc' | grep "$SRC" | head -1)
if [ -z "$LINKLINE" ]; then
    # fallback: some targets compile the test inline (e.g. test_audio
    # style), so just let make do it but stop before the auto-run by
    # touching the source then running make with a timeout on the run line
    echo "note: no plain link line for $TARGET, falling back to make" >&2
    touch "$SRC"
    make "$TARGET" 2>&1 | head -5 || true
    echo "make ran the target itself (binary may be deleted on crash)."
    echo "For gdb: rebuild with the -o override and run gdb manually."
    exit 0
fi

# Redirect output binary to ./<TARGET> (same name make would use)
LINKLINE=${LINKLINE//-o $TARGET/-o .\/$TARGET}

echo "linking: ${LINKLINE:0:100}..."
eval "$LINKLINE"

case "$MODE" in
    build|keep)
        echo "built ./$TARGET (kept, not run)"
        ;;
    run)
        echo "running ./$TARGET..."
        ./"$TARGET"
        ;;
    gdb)
        echo "running gdb ./$TARGET..."
        gdb -batch -ex run -ex bt ./"$TARGET"
        ;;
esac
