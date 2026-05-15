#!/bin/bash
# Simple sequential regression — 3 prompts, GPU for ours
set -e
MODEL="/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
LLAMA="/home/wubu/llama.cpp/build/bin/llama-cli"
cd /home/wubu/bytropix

echo "=== Quick Regression (3 prompts) ==="
PASS=0; TOTAL=0

for prompt in "Hello world" "The capital of France is" "2+2="; do
    TOTAL=$((TOTAL + 1))
    echo "--- [$TOTAL] \"$prompt\" ---"
    
    # Our engine — GPU, 4 tokens, 30s timeout
    OUR=$(timeout 30 ./infer_text "$MODEL" "$prompt" 4 20 2>/dev/null | grep "Top-5" | head -1)
    echo "  Ours: ${OUR:-TIMEOUT}"
    
    # llama — 4 tokens, 60s timeout
    LLM=$(timeout 60 bash -c 'echo -n "$1" | '"$LLAMA"' -m "$MODEL" -n 4 -t 24 --temp 1.0 --top-k 20 --top-p 0.95 --no-display-prompt 2>/dev/null' _ "$prompt" | tr -s '\n' ' ' | grep -oP '[A-Za-z].*?(?=\[|$)' | head -1)
    echo "  llama: ${LLM:-TIMEOUT}"
    
    [ -n "$OUR" ] && PASS=$((PASS + 1)) && echo "  ✅" || echo "  ❌"
    echo ""
done

echo "=== Result: $PASS/$TOTAL producing output ==="
[ $PASS -gt 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
