#!/bin/bash
# Fast regression: 5 prompts, parallel our vs llama
set -e

MODEL="/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
LLAMA="/home/wubu/llama.cpp/build/bin/llama-cli"
OUR="./infer_text"
cd /home/wubu/bytropix

PROMPTS=(
    "Hello world"
    "The capital of France is"
    "2 + 2 ="
    "Translate to French: Hello"
    "What is AI?"
)

echo "=== Fast Regression (5 prompts, parallel) ==="
PASS=0; TOTAL=0

for prompt in "${PROMPTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "--- [$TOTAL/$TOTAL] \"$prompt\" ---"
    
    # Run both in parallel
    OUR_OUT=$(NOGPU=1 MOE=0 timeout 30 $OUR "$MODEL" "$prompt" 4 20 2>/dev/null | grep "Top-5" | head -1) &
    OUR_PID=$!
    
    LLAMA_OUT=$(echo -n "$prompt" | timeout 60 $LLAMA -m "$MODEL" -n 4 -t 24 --temp 1.0 --top-k 20 --top-p 0.95 --no-display-prompt 2>/dev/null | tr '\r\n' '\n' | grep -v "^$\|^> \|▄\|██\|▀▀\|build\|model \|modali\|availab\|/exit\|  /" | tr -d '\b' | head -1) &
    LLAMA_PID=$!
    
    wait $OUR_PID 2>/dev/null; wait $LLAMA_PID 2>/dev/null
    
    echo "  Ours:  ${OUR_OUT:-"(timeout)"}"
    echo "  llama: ${LLAMA_OUT:-"(timeout)"}"
    
    if [ -n "$OUR_OUT" ]; then
        echo "  ✅ runs"
        PASS=$((PASS + 1))
    else
        echo "  ❌ no output"
    fi
    echo ""
done

echo "=== Result: $PASS/$TOTAL producing output ==="
[ $PASS -gt 0 ] && echo "✅ PASS" || { echo "❌ FAIL"; exit 1; }
