#!/usr/bin/env bash
# audit_c11.sh — measure the strict-C11 gap over the whole CORE_OBJ build.
#
# Goal #773 wants `gcc -std=c11 -pedantic -fsyntax-only` over every CORE_OBJ
# source to produce an empty diagnostic file. It does not, yet. This script
# produces the honest number instead of a claim:
#   - maps .o -> .c, skips .cu/.S (they are not C11 translation units)
#   - counts diagnostics, errors, and the files that own them
#   - exits 0 when clean, 1 when not (so it can become a gate later)
#
# Usage: bash tools/audit_c11.sh [--list-errors]
set -u
cd "$(dirname "$0")/.." || exit 2

OUT="${TMPDIR:-/tmp}/wubu_c11_audit.$$"
LIST="$OUT.srcs"
DIAG="$OUT.diag"
trap 'rm -f "$LIST" "$DIAG" "$OUT".*' EXIT

make -pn 2>/dev/null | awk -F'= ' '/^CORE_OBJ *=/{print $2}' \
  | tr ' ' '\n' | sed 's/\.o$/.c/' | grep '^src/' | sort -u > "$LIST"

# .cu and .S are not C11 translation units
if [ -s "$LIST" ]; then
  grep -vE '\.(cu|S)$' "$LIST" > "$LIST.c" && mv "$LIST.c" "$LIST"
fi
N=$(wc -l < "$LIST")

: > "$DIAG"
while IFS= read -r f; do
  [ -f "$f" ] || { echo "$f: MISSING SOURCE" >> "$DIAG"; continue; }
  gcc -std=c11 -pedantic -fsyntax-only -I include "$f" 2>> "$DIAG" || true
done < "$LIST"

DIAG_LINES=$(wc -l < "$DIAG")
ERRORS=$(grep -c 'error:' "$DIAG" || true)
FILES=$(grep -oE '^[^:]+\.c' "$DIAG" | sort -u | wc -l)

echo "CORE_OBJ C sources:      $N"
echo "diagnostic lines:        $DIAG_LINES"
echo "errors:                  $ERRORS"
echo "files with diagnostics:  $FILES"

if [ "${1:-}" = "--list-errors" ]; then
  echo "--- errors ---"
  grep 'error:' "$DIAG" | sort -u
fi

[ "$DIAG_LINES" -eq 0 ]
