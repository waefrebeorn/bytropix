#!/bin/sh
# test_cycle.sh -- prove one whole Harness^2 cycle with a fake editor.
#
# No model is involved: the editor is a shell command that prints a bullet list,
# and the check reads the harness the cycle staged. Adoption therefore happens
# because the rule reached the staged file, not because anything claimed success.
set -eu

ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT

cat > "$ROOT/editor.sh" <<EOF
#!/bin/sh
printf -- '- rule X\n'
EOF
chmod +x "$ROOT/editor.sh"

CHECK="grep -q 'rule X' $ROOT/run/cand-0/AGENTS.md"

set +e
./h2_cycle --task t1 --check "$CHECK" --root "$ROOT/run" \
           --editor "sh $ROOT/editor.sh" --timeout 10 > "$ROOT/out.txt" 2>&1
rc=$?
set -e

echo "--- cycle output ---"
cat "$ROOT/out.txt"

fail=0
[ "$rc" -eq 0 ] || { echo "FAIL: exit code $rc (expected 0)"; fail=1; }
grep -q 'baseline: check_rc=[1-9]' "$ROOT/out.txt" || { echo "FAIL: baseline did not fail"; fail=1; }
grep -q 'compose: adopted probe 1' "$ROOT/out.txt" || { echo "FAIL: candidate not adopted"; fail=1; }
[ -f "$ROOT/run/LEDGER.jsonl" ] || { echo "FAIL: no ledger"; fail=1; }
grep -q '"adopted":1' "$ROOT/run/LEDGER.jsonl" || { echo "FAIL: ledger has no adoption"; fail=1; }
[ -f "$ROOT/run/library/t1/EXPERIENCE.md" ] || { echo "FAIL: no library entry"; fail=1; }
grep -q 'cue: rule X' "$ROOT/run/library/t1/EXPERIENCE.md" || { echo "FAIL: lesson cue missing"; fail=1; }

# Negative control: a check that never passes must adopt nothing and exit 1.
set +e
./h2_cycle --task t2 --check 'test -f /nonexistent-h2' --root "$ROOT/run2" \
           --editor "sh $ROOT/editor.sh" --timeout 10 > "$ROOT/out2.txt" 2>&1
rc2=$?
set -e
[ "$rc2" -eq 1 ] || { echo "FAIL: impossible check exited $rc2 (expected 1)"; fail=1; }
grep -q '"adopted":-1' "$ROOT/run2/LEDGER.jsonl" || { echo "FAIL: adopted something that never passed"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "=== test_cycle: ALL PASSED ==="
else
    echo "=== test_cycle: FAILED ==="
fi
exit "$fail"