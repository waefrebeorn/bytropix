#!/bin/bash
# check_dupes.sh -- STACK #362 duplicate-symbol gate.
#
# Detects duplicate GLOBAL symbol definitions (T/D/B/R) across a set of
# object files that are meant to be co-linked. A duplicate means two .o files
# both define the same global -> a link-time multiple-definition error (or,
# worse, silent ODR-style wrong-symbol resolution).
#
# Usage: check_dupes.sh <obj1.o> <obj2.o> ...
#   Exits 0 if no duplicates, 1 if any (printing them).
#
# Intentionally-excluded pairs (legacy aggregate vs split module, real vs
# stub) are NEVER co-linked in one target, so this gate is always run on a
# single target's object list -- never on the whole src/ tree.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: check_dupes.sh <obj> [<obj>...]" >&2
  exit 2
fi

# Collect defined global symbols per object, then find any defined >1 time.
# nm output: "<addr> <type> <name>"; we keep T/D/B/R (global defs) and ignore
# W/V (weak/undefined) and local (t/d/b/r).
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

for o in "$@"; do
  [ -f "$o" ] || { echo "check_dupes: missing object $o" >&2; exit 2; }
  nm --defined-only "$o" 2>/dev/null \
    | awk '$2=="T"||$2=="D"||$2=="B"||$2=="R"{print $3"\t"FILENAME}' \
    | sed "s|$|\t$o|"
done | sort > "$tmp"

# symbols appearing more than once (defined in 2+ of the given objects)
dupes="$(awk -F'\t' '{print $1}' "$tmp" | sort | uniq -d)"
if [ -n "$dupes" ]; then
  echo "check_dupes: DUPLICATE global symbol definitions:" >&2
  echo "$dupes" | while read -r sym; do
    echo "  $sym:" >&2
    grep -P "^\Q$sym\E\t" "$tmp" | awk -F'\t' '{print "    "$3}' >&2
  done
  exit 1
fi
echo "check_dupes: no duplicate symbols across $# object(s)"
exit 0
