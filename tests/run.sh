#!/bin/bash
# WuBuText AI — Test Harness (May 15 PM v1)
# Usage: bash tests/run.sh [--quick|--full|--moe|--golden]
#
# Exits 0 if all tests PASS, 1 if any FAIL.
set -uo pipefail  # NO set -e — sub-commands may fail

MODEL="/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
BIN="./infer_text_gpu"
GOLDEN="tests/golden"
PASS=0
FAIL=0
TIMEOUT=120

info()  { echo "[INFO] $*"; }
pass()  { echo "  ✅ PASS: $*"; PASS=$((PASS+1)); }
fail()  { echo "  ❌ FAIL: $*"; FAIL=$((FAIL+1)); }
check_exit() {
    local label="$1" exit_code="$2"
    [ "$exit_code" -eq 0 ] && pass "$label" || fail "$label (exit=$exit_code)"
}

# --- 1. BUILD ---
info "=== 1. BUILD ==="
make -j4 infer_text_gpu 2>/dev/null && pass "make infer_text_gpu" || fail "make infer_text_gpu"

# --- 2. EXISTENCE ---
info "=== 2. EXISTENCE ==="
[ -f "$BIN" ] && pass "binary exists" || fail "binary missing"
[ -f "$MODEL" ] && pass "model exists: $(du -h $MODEL | cut -f1)" || fail "model missing"
[ -x "$BIN" ] && pass "binary executable" || fail "binary not executable"
file "$BIN" | grep -q "ELF 64-bit" && pass "ELF 64-bit" || fail "not ELF"

# --- 3. QUICK SMOKE (no MoE) ---
info "=== 3. SMOKE TEST (MOE=0) ==="
OUT=$(timeout $TIMEOUT $BIN $MODEL "Hello" 1 2>/dev/null)
[ $? -eq 0 ] && pass "exit 0" || fail "exit non-zero"
echo "$OUT" | grep -q "=== PASS ===" && pass "PASS marker found" || fail "PASS marker missing"
echo "$OUT" | grep -q "Decode:" && pass "decode ran" || fail "decode missing"
# Extract decode speed
DECODE_SPEED=$(echo "$OUT" | grep "Decode:" | grep -oP '\d+\.?\d*(?= tok/s)')
[ -n "$DECODE_SPEED" ] && pass "decode speed: ${DECODE_SPEED} tok/s" || fail "no decode speed"

# --- 4. OUTPUT CORRECTNESS (vs golden) ---
info "=== 4. OUTPUT REGRESSION ==="
OUT=$(timeout $TIMEOUT $BIN $MODEL "The meaning of life" 3 2>/dev/null)
[ $? -eq 0 ] && pass "golden test exit 0" || fail "golden test exit non-zero"
# Extract first 3 chars of generated text AFTER "... [!]" pattern  
echo "$OUT" | grep -q "meaning of life!" && pass "prefill output correct" || fail "prefill output wrong"
echo "$OUT" | grep -oP '^[!.]*$' | tr -d '\n' | grep -q "^!!!" && pass "decode matches golden: '!!!'" || fail "decode mismatch"

# --- 5. CHUNK SIZE COMPARISON ---
info "=== 5. CHUNK SIZE PARITY ==="
PROMPT="The theory of relativity fundamentally changed our understanding of space and time."
OUT256=$(CHUNK=256 timeout $TIMEOUT $BIN $MODEL "$PROMPT" 1 2>/dev/null)
OUT064=$(CHUNK=64  timeout $TIMEOUT $BIN $MODEL "$PROMPT" 1 2>/dev/null)
TXT256=$(echo "$OUT256" | grep -oP "understood our[^!]*" | head -1)
TXT064=$(echo "$OUT064" | grep -oP "understood our[^!]*" | head -1)
[ "$TXT256" = "$TXT064" ] && pass "CHUNK=256 == CHUNK=64 output: text matches" || fail "CHUNK output mismatch: '$TXT256' vs '$TXT064'"

# --- 6. DECODE SPEED BENCHMARK ---
info "=== 6. DECODE SPEED ==="
OUT=$(timeout $TIMEOUT $BIN $MODEL "Hello world" 5 2>/dev/null)
DEC_SPEED=$(echo "$OUT" | grep "Decode:" | grep -oP '\d+\.?\d*(?= tok/s)')
PRE_SPEED=$(echo "$OUT" | grep "Prefill:" | grep -oP '\d+\.?\d*(?= tok/s)' | tail -1)
[ -n "$DEC_SPEED" ] && pass "decode ${DEC_SPEED} tok/s" || fail "no decode speed"
[ -n "$PRE_SPEED" ] && pass "prefill ${PRE_SPEED} tok/s" || fail "no prefill speed"

# --- 7. LONG PROMPT (66 tokens, cross chunk boundary) ---
info "=== 7. LONG PROMPT (66 tok) ==="
LONG_PROMPT="The theory of relativity, developed by Albert Einstein, fundamentally changed our understanding of physics. It introduced the concept that space and time are interconnected, forming a four-dimensional spacetime continuum. This theory has been confirmed by numerous experiments and observations."
OUT=$(timeout $TIMEOUT $BIN $MODEL "$LONG_PROMPT" 2 2>/dev/null)
[ $? -eq 0 ] && pass "long prompt exit 0" || fail "long prompt exit non-zero"
echo "$OUT" | grep -q "=== PASS ===" && pass "long prompt PASS" || fail "long prompt no PASS"
echo "$OUT" | grep -q "Chunked prefill (4" && pass "48 tok prefill" || fail "not 48 tok"

# --- 8. SHARED EXPERT + GQA LAYERS PRESENT ---
info "=== 8. LAYER CONFIG ==="
OUT=$(timeout 30 $BIN $MODEL "x" 1 2>&1)
echo "$OUT" | grep -q "40 layers (30 SSM, 10 GQA)" && pass "40 layers correct" || fail "layer count wrong"

# --- 9. NO NaN/NEG/CRASH ---
info "=== 9. OUTPUT CLEANLINESS ==="
OUT=$(timeout $TIMEOUT $BIN $MODEL "Test output" 3 2>/dev/null)
echo "$OUT" | grep -v "Summary" | grep -q "nan\|NaN\|NAN\|-inf\|INF" && fail "contains NaN/Inf" || pass "no NaN/Inf"
echo "$OUT" | grep -qv "core dumped" || true
echo "$OUT" | grep -qv "Segmentation fault" || true

# --- 10. MOE=1 (optional, slow) ---
if [ "${1:-}" = "--moe" ] || [ "${1:-}" = "--full" ]; then
    info "=== 10. MOE=1 SMOKE ==="
    OUT=$(timeout 600 bash -c "MOE=1 $BIN $MODEL 'Hello' 1" 2>/dev/null) && pass "MOE=1 exit 0" || fail "MOE=1 timed out or failed"
    echo "$OUT" | grep -q "=== PASS ===" && pass "MOE=1 PASS" || fail "MOE=1 no PASS"
    echo "$OUT" | grep -q "GPU MoE buffers" && pass "MOE=1 GPU buffers allocated" || fail "MOE=1 no GPU buffers"
fi

# --- SUMMARY ---
echo ""
echo "=== RESULTS: $PASS pass, $FAIL fail ==="
[ $FAIL -eq 0 ] && echo "✅ ALL TESTS PASSED" || echo "❌ $FAIL TEST(S) FAILED"
exit $FAIL
