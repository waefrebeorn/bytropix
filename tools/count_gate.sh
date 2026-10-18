#!/bin/bash
# Run the wubuwizard gate HONESTLY: every test target must execute.
# Make skips a target whose executable is newer than its sources, so a
# previous run can make the gate look green while running almost nothing.
# Deleting the top-level test binaries forces every recipe to build+run.
cd /home/wubu/wubuwizard || exit 1
find . -maxdepth 1 -type f -name 'test_*' ! -name '*.c' -delete
make test_all > /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log 2>&1
rc=$?
echo "EXIT=$rc"
echo "binaries run: $(grep -c '^\./test_' /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log)"
echo "ALL-PASSED lines: $(grep -c 'ALL .*PASSED\|ALL TESTS PASSED' /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log)"
echo "FAIL lines: $(grep -cE 'FAIL|NOT PASSED|Error 1' /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log)"
echo "--- last lines ---"
tail -5 /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log
exit $rc
