#!/usr/bin/env bash
# Sandbox API Test Suite
set -euo pipefail

PORT=8089
BASE="http://localhost:${PORT}"
KEY="sk-sandbox-test-key-1"
PASS=0; FAIL=0

banner() { echo -e "\n=== $1 ==="; }
pass() { PASS=$((PASS+1)); echo "  ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL $1"; }

cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null || true; }
trap cleanup EXIT

start_server() {
    cd /home/wubu/bytropix
    python3 tools/serve.py --sandbox --port "$PORT" > /tmp/api_test.log 2>&1 &
    PID=$!
    for i in $(seq 1 20); do
        curl -sf "$BASE/health" > /dev/null 2>&1 && return 0
        sleep 0.5
    done
    cat /tmp/api_test.log; return 1
}

req() {
    local method="$1" path="$2" body="$3"
    curl -s -X "$method" "$BASE$path" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $KEY" \
        -d "$body" 2>/dev/null || true
}

nokey_req() {
    local method="$1" path="$2" body="$3" akey="$4"
    curl -s -X "$method" "$BASE$path" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $akey" \
        -d "$body" 2>/dev/null || true
}

# --- Tests ---

test_health() {
    banner "Health"
    local r=$(curl -sf "$BASE/health" 2>/dev/null)
    echo "$r" | grep -q '"status": "ok"' && pass "health ok" || fail "health: $r"
    echo "$r" | grep -q '"mode": "sandbox"' && pass "sandbox mode" || fail "not sandbox: $r"
}

test_models() {
    banner "Models"
    local r=$(curl -sf "$BASE/v1/models" 2>/dev/null)
    echo "$r" | grep -q "qwen3.6" && pass "models list" || fail "models: $r"
}

test_completions() {
    banner "Completions"
    local r=$(req POST /v1/completions '{"prompt":"Hello","max_tokens":10}')
    echo "$r" | grep -q '"text"' && pass "returns text" || fail "completion: $r"
    echo "$r" | grep -q '"usage"' && pass "usage stats" || fail "no usage: $r"
}

test_chat() {
    banner "Chat"
    local r=$(req POST /v1/chat/completions '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}')
    echo "$r" | grep -q '"content"' && pass "chat content" || fail "chat: $r"
    echo "$r" | grep -q '"assistant"' && pass "assistant role" || fail "role: $r"
}

test_auth() {
    banner "Auth"
    local r=$(nokey_req POST /v1/completions '{"prompt":"test"}' "invalid-key")
    echo "$r" | grep -q "authentication_error" && pass "bad key rejected" || fail "bad key: $r"
}

test_ratelimit() {
    banner "Rate Limit"
    local rk="sk-sandbox-ratelimit"
    for i in 1 2; do
        nokey_req POST /v1/completions '{"prompt":"test","max_tokens":5}' "$rk" > /dev/null
    done
    pass "two requests consumed"
    local r=$(nokey_req POST /v1/completions '{"prompt":"test","max_tokens":5}' "$rk")
    echo "$r" | grep -q "rate_limit_error" && pass "third blocked 429" || fail "not blocked: $r"
}

test_fuzz() {
    banner "Fuzzing"
    r=$(req POST /v1/completions '')
    [ -n "$r" ] && pass "empty body" || fail "empty body crash"

    r=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d 'not json' 2>/dev/null || true)
    echo "$r" | grep -q "error" && pass "bad json" || fail "bad json: $r"

    r=$(req POST /v1/completions '{"max_tokens":5}')
test_404() {
    banner "404"
    local r=$(req POST /v1/unknown '{"prompt":"test"}')
    echo "$r" | grep -q "Unknown endpoint" && pass "unknown endpoint" || fail "unknown: $r"
}

test_stream() {
    banner "Streaming"
    local r=$(timeout 5 bash -c "curl -s -X POST 'http://localhost:${PORT}/v1/completions' -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-sandbox-test-key-1' -d '{\"prompt\":\"Hi\",\"max_tokens\":5,\"stream\":true}'" 2>/dev/null || true)
    echo "$r" | grep -q "data:" && pass "SSE data events" || fail "no SSE"
    echo "$r" | grep -q 'DONE' && pass "stream ends" || fail "no DONE"
}

test_cors() {
    banner "CORS"
    local r=$(curl -sfI -X OPTIONS "$BASE/v1/completions" 2>/dev/null)
    echo "$r" | grep -qi "access-control-allow-origin" && pass "CORS headers" || fail "no CORS: $r"
}

test_404() {
    banner "404"
    local r=$(req POST /v1/unknown '{"prompt":"test"}')
    echo "$r" | grep -q "Not found" && pass "unknown endpoint" || fail "unknown: $r"
}

# --- Main ---

echo "=== API Server Sandbox Test Suite ==="
start_server
echo "Server PID=$PID"

test_health
test_models
test_completions
test_chat
test_auth
test_cors
test_404
test_stream
test_ratelimit
test_fuzz

echo ""
echo "=== Results: $PASS pass, $FAIL fail ==="
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
