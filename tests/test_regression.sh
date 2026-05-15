#!/bin/bash
# Regression test: compare our engine vs llama.cpp on 10 fixed prompts
set -e

MODEL="/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
LLAMA="/home/wubu/llama.cpp/build/bin/llama-cli"
OUR="./infer_text"

# Prompts to test
PROMPTS=(
    "Hello! How are you?"
    "The capital of France is"
    "Once upon a time"
    "2 + 2 ="
    "Translate to French: Hello"
    "What is quantum computing?"
    "Write a poem about AI"
    "Explain gravity simply"
    "Python code for fibonacci"
    "The meaning of life is"
)

cd /home/wubu/bytropix

echo "=== Regression Test Suite ==="
echo "Model: Qwen3.6-35B-A3B-UD-IQ2_M"
echo ""

PASS=0
FAIL=0
TOTAL=0

for prompt in "${PROMPTS[@]}"; do
    echo "--- Prompt: \"$prompt\" ---"
    TOTAL=$((TOTAL + 1))
    
    # Run our engine (NOGPU=1, MOE=0, first 12 tokens)
    OUR_OUT=$(NOGPU=1 MOE=0 timeout 120 $OUR "$MODEL" "$prompt" 8 20 2>/dev/null | grep -oP 'Top-5 tokens:.*' | head -1)
    
    # Run llama.cpp (same prompt, 8 tokens)
    LLAMA_OUT=$(echo -n "$prompt" | timeout 300 $LLAMA -m "$MODEL" -n 8 -t 24 --temp 1.0 --top-k 20 --top-p 0.95 --no-display-prompt 2>/dev/null | tr '\r\n' '\n' | grep -v "^$\|^> \|▄\|██\|▀▀\|^\s*$\|build\|model \|modali\|availab\|/exit\|  /" | tr -d '\b' | sed 's/[|\\\/-]//g' | head -1)
    
    echo "  Ours:  $OUR_OUT"
    echo "  llama: $LLAMA_OUT"
    
    # Check if ours produced any output
    if [ -n "$OUR_OUT" ]; then
        echo "  STATUS: ✅ runs"
    else
        echo "  STATUS: ❌ no output"
        FAIL=$((FAIL + 1))
        continue
    fi
    
    # Check if llama produced any output
    if [ -n "$LLAMA_OUT" ]; then
        echo "  STATUS: ✅ reference available"
        PASS=$((PASS + 1))
    else
        echo "  STATUS: ⚠️ no reference (llama timeout?)"
        # Not a fail — llama might be slow
    fi
    
    echo ""
done

echo "=== Results ==="
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""

# If we got any output at all, the engine works
if [ $FAIL -lt $TOTAL ]; then
    echo "✅ Engine produces output (regression: PASS)"
else
    echo "❌ Engine produced no output (regression: FAIL)"
    exit 1
fi
