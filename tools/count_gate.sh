#!/bin/bash
# Count PASS lines from the wubuwizard gate run
cd /home/wubu/wubuwizard
make test_all > /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log 2>&1
echo "EXIT=$?"
echo "PASS lines: $(grep -c '\[PASS\]' /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log)"
echo "FAIL lines: $(grep -cE '\[FAIL\]|FAILED|NOT PASSED' /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log)"
echo "--- result lines ---"
grep -E "ALL PASSED|FAILED|NOT PASSED|Error" /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log | head -30
echo "--- test binaries run ---"
grep -c "^\./test_" /home/wubu/.hermes/profiles/mind-palace/cache/wz_gate.log
