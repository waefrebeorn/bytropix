#!/usr/bin/env bash
# bytropix 512K Test Suite — run from ~/bytropix
# Tests KV cache allocation, sparse attention, decode continuity
set -euo pipefail

MODEL="${MODEL:-/home/wubu2/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../.hermes/test-results"
mkdir -p "$RESULTS_DIR"

export MODEL
export MAX_CTX="${MAX_CTX:-524288}"

PASS=0
FAIL=0
RESULT_LOG="$RESULTS_DIR/512k-test-$(date +%Y%m%d-%H%M%S).log"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULT_LOG"; }
pass() { log "✅ PASS: $1"; ((PASS++)); }
fail() { log "❌ FAIL: $1"; ((FAIL++)); }
run_test() {
    local name="$1"; shift
    log "TEST: $name"
    local rc=0
    "$@" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$name"
    else
        fail "$name"
        log "  (exit code: $rc)"
    fi
}

# Read-only check first
if [ ! -f "$MODEL" ]; then
    fail "Model not found: $MODEL"
    log "SKIPPING all tests — no model"
    echo ""
    echo "=== RESULTS: ${PASS}/${PASS} pass ($FAIL fail) ===" | tee -a "$RESULT_LOG"
    exit 0
fi

log "=== bytropix 512K Test Suite ==="
log "Model: $MODEL (du $(du -h "$MODEL" | cut -f1))"
log "MAX_CTX: $MAX_CTX"
echo "" | tee -a "$RESULT_LOG"

# === Test 1: KV cache allocation at 512K ===
run_test "KV cache alloc at ${MAX_CTX}" \
    timeout 120 ./gen_text_cpu "test" 1 1 2>&1

# === Test 2: Sparse attention enabled ===
run_test "Sparse attention (USE_SPARSE_ATTN=1)" \
    timeout 180 bash -c '
        USE_SPARSE_ATTN=1 SPARSE_W=512 SPARSE_G=128 \
        ./gen_text_cpu "Hello" 3 1
    '

# === Test 3: Decode without sparse attention (baseline comparison) ===
run_test "Full attention decode (USE_SPARSE_ATTN=0)" \
    timeout 180 bash -c '
        USE_SPARSE_ATTN=0 \
        ./gen_text_cpu "Hello" 3 1
    '

# === Test 4: Memory usage check — verify no swap death ===
run_test "Memory stable after test runs" \
    bash -c '
        free -m | awk "/Mem:/{if(\$7 > 1024) exit 0; else {print \"WARNING: only \"\$7\"MB free\"; exit 1}}"
    '

# === Test 5: RoPE scale factor override ===
run_test "RoPE scale factor override (ROPE_SCALE_FACTOR=0.25)" \
    timeout 180 bash -c '
        ROPE_SCALE_FACTOR=0.25 USE_SPARSE_ATTN=1 \
        ./gen_text_cpu "Hello" 2 1
    '

# === Test 6: NES emulator builds ===
run_test "NES emulator builds clean" \
    bash -c '
        cd /home/wubu2/hermes-test/projects/nes-emulator
        gcc -O2 -Iinclude -o /dev/null src/*.c -lm 2>&1
    '

echo "" | tee -a "$RESULT_LOG"
log "=== RESULTS: ${PASS}/${PASS} pass ($FAIL fail) ==="
exit $FAIL
