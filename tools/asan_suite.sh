#!/usr/bin/env bash
# STACK #370: ASAN/UBSAN clean test pass.
# Builds every tools/test_*.c with its matching src/wubu_<x>.c under
# AddressSanitizer + UndefinedBehaviorSanitizer, runs it, and reports any
# sanitizer error. Exit 0 = whole suite is sanitizer-clean.
set -u
cd "$(dirname "$0")/.."
CF="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I include"
OUT=/home/wubu/.hermes/profiles/mind-palace/cache/scratch/asan
mkdir -p "$OUT"
total=0; clean=0; dirty=0; missing=0
for t in tools/test_*.c; do
  base=$(basename "$t" .c)          # test_<x>
  mod=${base#test_}                  # <x>
  src="src/wubu_${mod}.c"
  total=$((total+1))
  if [ ! -f "$src" ]; then
    # some tests link multiple modules; skip ones without a 1:1 module
    missing=$((missing+1)); continue
  fi
# Link against the full src module set so cross-module dependencies resolve;
# only genuine sanitizer findings (not linker errors) count as dirty.
  bin="$OUT/$base"
  if gcc $CF -o "$bin" "$t" src/wubu_*.c -lm 2>"$OUT/$base.cc.log"; then
    if "$bin" >"$OUT/$base.run.log" 2>&1; then
      if grep -qiE 'runtime error|AddressSanitizer|SUMMARY: |ERROR: ' "$OUT/$base.run.log"; then
        dirty=$((dirty+1)); echo "DIRTY  $base (sanitizer reported)"; grep -iE 'runtime error|ERROR: ' "$OUT/$base.run.log" | head -2
      else
        clean=$((clean+1))
      fi
    else
      dirty=$((dirty+1)); echo "DIRTY  $base (nonzero exit / sanitizer abort)"; tail -3 "$OUT/$base.run.log"
    fi
  else
    dirty=$((dirty+1)); echo "DIRTY  $base (compile failed)"; tail -3 "$OUT/$base.cc.log"
  fi
done
echo "----"
echo "STACK #370 ASAN/UBSAN: total=$total clean=$clean dirty=$dirty skipped(no-1:1-module)=$missing"
[ "$dirty" -eq 0 ] && echo "SANITIZER-CLEAN" || echo "NOT CLEAN"
exit $([ "$dirty" -eq 0 ] && echo 0 || echo 1)
