#!/bin/bash
# run_512k_full.sh - Runs the 512k Comprehensive Benchmark Suite

set -e

# Defaults
MODEL="${1:-/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}"
CONTEXT="${2:-524288}"          # 512k default
TRIALS="${3:-3}"
BENCH_TYPE="${4:-6}"            # 6=ALL

BUILD_DIR="/home/wubu/bytropix/build"
BENCH_BIN="$BUILD_DIR/bench_512k_full"

echo "============================================================"
echo "  512k Comprehensive Context Benchmark"
echo "============================================================"
echo "Model: $MODEL"
echo "Context: $CONTEXT (${CONTEXT}/1024 k)"
echo "Trials: $TRIALS"
echo "Benchmark: ${BENCH_TYPE} (0=NIAH, 1=RULER, 2=LongCode, 3=AgentLong, 4=LongBench-v2, 5=MIR, 6=ALL)"
echo ""

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

# Build if needed
if [[ ! -f "$BENCH_BIN" ]]; then
    echo "Building benchmark..."
    /home/wubu/bytropix/tools/build_bench.sh
fi

if [[ ! -f "$BENCH_BIN" ]]; then
    echo "ERROR: Build failed, binary not found"
    exit 1
fi

# Create log directory
LOG_DIR="$BUILD_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/bench_512k_full_${CONTEXT}k_$(date +%Y%m%d_%H%M%S).log"

echo "Running full benchmark suite..."
echo "Log: $LOG_FILE"
echo ""

# Run with GPU if available and context <= 256k
GPU_ENV=""
if [[ "$CONTEXT" -le 262144 ]]; then
    GPU_ENV="GPU=1"
fi

# Execute
if [[ -n "$GPU_ENV" ]]; then
    time env $GPU_ENV "$BENCH_BIN" "$MODEL" "$CONTEXT" "$TRIALS" "$BENCH_TYPE" 2>&1 | tee "$LOG_FILE"
else
    time "$BENCH_BIN" "$MODEL" "$CONTEXT" "$TRIALS" "$BENCH_TYPE" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "============================================================"
echo "  Benchmark Complete - Log saved to $LOG_FILE"
echo "============================================================"