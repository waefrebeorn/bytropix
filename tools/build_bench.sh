#!/bin/bash
# build_bench.sh - Build the 512k Comprehensive Benchmark

set -e

# Paths
SRC_DIR="/home/wubu/bytropix"
BUILD_DIR="$SRC_DIR/build"
TOOLS_DIR="$SRC_DIR/tools"
SRC="$SRC_DIR/src"
INC="$SRC_DIR/include"

CC="gcc"
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CFLAGS="-O3 -march=native -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -I$INC"
NVCC_FLAGS="-O3 -std=c++17 -I$INC -arch=native --ptxas-options=-v -DGPU_SUPPORT -DDEF_N_EXPERTS=256 -DDEF_N_ACTIVE_EXPTS=8 -DDEF_D_FF=512 -DDEF_SHARED_D_FF=512 -DDEF_D_MODEL=2048"
LDFLAGS="-pthread -L/usr/local/cuda-13.1/lib64 -L/usr/lib/wsl/lib -lcublas -lcudart -lcuda -lcurand -lstdc++ -fopenmp"
LIBM="-lm -ldl"

echo "Building 512k Comprehensive Benchmark..."
echo "Source: $TOOLS_DIR/bench_512k_full.c"
mkdir -p "$BUILD_DIR"

# Check for required source files
REQUIRED_FILES=(
    "$SRC/wubu_model.c"
    "$SRC/wubu_tokenizer.c"
    "$SRC/gguf_reader.c"
    "$SRC/gaad_nesting_llm.c"
    "$SRC/wubu_mobius.c"
    "$SRC/kv_paged_attention.c"
    "$SRC/cuda_kernels.cu"
    "$SRC/flash_attn_q4_0_opt.cu"
    "$SRC/wubu_model_gpu.cu"
    "$SRC/wubu_ssm.c"
    "$SRC/quantized_dot_generic.c"
    "$SRC/rotorquant.cu"
    "$SRC/poincare_attn.cu"
    "$SRC/kv_arena.cu"
    "$SRC/gpu_output_proj.cu"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "[WARN] Missing: $f"
    fi
done

# Compile C files
$CC $CFLAGS -c "$SRC/wubu_model.c" -o "$BUILD_DIR/wubu_model.o"
$CC $CFLAGS -c "$SRC/wubu_tokenizer.c" -o "$BUILD_DIR/wubu_tokenizer.o"
$CC $CFLAGS -c "$SRC/gguf_reader.c" -o "$BUILD_DIR/gguf_reader.o"
$CC $CFLAGS -c "$SRC/gaad_nesting_llm.c" -o "$BUILD_DIR/gaad_nesting_llm.o"
$CC $CFLAGS -c "$SRC/wubu_mobius.c" -o "$BUILD_DIR/wubu_mobius.o"
$CC $CFLAGS -c "$SRC/quantized_dot_generic.c" -o "$BUILD_DIR/quantized_dot_generic.o"
$CC $CFLAGS -c "$SRC/wubu_ssm.c" -o "$BUILD_DIR/wubu_ssm.o"
$CC $CFLAGS -c "$SRC/wubu_ssm_chunked.c" -o "$BUILD_DIR/wubu_ssm_chunked.o"
$CC $CFLAGS -c "$SRC/wubu_poincare_gqa.c" -o "$BUILD_DIR/wubu_poincare_gqa.o"
$CC $CFLAGS -c "$SRC/wubu_poincare_gqa_backward.c" -o "$BUILD_DIR/wubu_poincare_gqa_backward.o"
$CC $CFLAGS -c "$SRC/wubu_poincare_ssm_backward.c" -o "$BUILD_DIR/wubu_poincare_ssm_backward.o"
$CC $CFLAGS -c "$SRC/wubu_nested_ssm.c" -o "$BUILD_DIR/wubu_nested_ssm.o"
$CC $CFLAGS -c "$SRC/wubu_nested_ssm_backward.c" -o "$BUILD_DIR/wubu_nested_ssm_backward.o"
$CC $CFLAGS -c "$SRC/wubu_moe.c" -o "$BUILD_DIR/wubu_moe.o"
# $CC $CFLAGS -c "$SRC/wubu_moe_hyperbolic.c" -o "$BUILD_DIR/wubu_moe_hyperbolic.o"
# $CC $CFLAGS -c "$SRC/wubu_moe_hyperbolic_backward.c" -o "$BUILD_DIR/wubu_moe_hyperbolic_backward.o"
# $CC $CFLAGS -c "$SRC/wubu_hyperbolic_output_proj.c" -o "$BUILD_DIR/wubu_hyperbolic_output_proj.o"
$CC $CFLAGS -c "$SRC/wubu_tst.c" -o "$BUILD_DIR/wubu_tst.o"
$CC $CFLAGS -c "$SRC/wubu_mobius_linear.c" -o "$BUILD_DIR/wubu_mobius_linear.o"
$CC $CFLAGS -c "$SRC/thread_pool.c" -o "$BUILD_DIR/thread_pool.o"
$CC $CFLAGS -c "$SRC/qlearner.c" -o "$BUILD_DIR/qlearner.o"
$CC $CFLAGS -c "$SRC/dequant_iq2_xxs.c" -o "$BUILD_DIR/dequant_iq2_xxs.o"
$CC $CFLAGS -c "$SRC/rsgd.c" -o "$BUILD_DIR/rsgd.o"
$CC $CFLAGS -c "$SRC/quantized_matmul.c" -o "$BUILD_DIR/quantized_matmul.o"

# Compile C files (including ones that need CUDA headers - compile with gcc + CUDA inc)
#$CC $CFLAGS -I/usr/local/cuda-13.1/include -c "$SRC/kv_paged_attention.c" -o "$BUILD_DIR/kv_paged_attention.o"

# Compile CUDA files
$NVCC $NVCC_FLAGS -c "$SRC/cuda_kernels.cu" -o "$BUILD_DIR/cuda_kernels.o"
$NVCC $NVCC_FLAGS -c "$SRC/flash_attn_q4_0_opt.cu" -o "$BUILD_DIR/flash_attn_q4_0_opt.o"
$NVCC $NVCC_FLAGS -c "$SRC/flash_attn_q4_0_prefill_opt.cu" -o "$BUILD_DIR/flash_attn_q4_0_prefill_opt.o"
$NVCC $NVCC_FLAGS -c "$SRC/wubu_model_gpu.cu" -o "$BUILD_DIR/wubu_model_gpu.o"
$NVCC $NVCC_FLAGS -c "$SRC/rotorquant.cu" -o "$BUILD_DIR/rotorquant.o"
$NVCC $NVCC_FLAGS -c "$SRC/poincare_attn.cu" -o "$BUILD_DIR/poincare_attn.o"
$NVCC $NVCC_FLAGS -c "$SRC/kv_arena.cu" -o "$BUILD_DIR/kv_arena.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_output_proj.cu" -o "$BUILD_DIR/gpu_output_proj.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_quant_matmul.cu" -o "$BUILD_DIR/gpu_quant_matmul.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_quant_matmul_row_major.cu" -o "$BUILD_DIR/gpu_quant_matmul_row_major.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_moe_kernel.cu" -o "$BUILD_DIR/gpu_moe_kernel.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_gemma4.cu" -o "$BUILD_DIR/gpu_gemma4.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_gemma4_forward.cu" -o "$BUILD_DIR/gpu_gemma4_forward.o"
$NVCC $NVCC_FLAGS -c "$SRC/gpu_ssm_recurrence.cu" -o "$BUILD_DIR/gpu_ssm_recurrence.o"

# Compile benchmark with gcc + CUDA includes
$CC $CFLAGS -I/usr/local/cuda-13.1/include -c "$TOOLS_DIR/bench_512k_full.c" -o "$BUILD_DIR/bench_512k_full.o"

# Link
$CC -o "$BUILD_DIR/bench_512k_full" \
    "$BUILD_DIR/bench_512k_full.o" \
    "$BUILD_DIR/wubu_model.o" \
    "$BUILD_DIR/wubu_tokenizer.o" \
    "$BUILD_DIR/gguf_reader.o" \
    "$BUILD_DIR/gaad_nesting_llm.o" \
    "$BUILD_DIR/wubu_mobius.o" \
    "$BUILD_DIR/quantized_dot_generic.o" \
    "$BUILD_DIR/wubu_ssm.o" \
    "$BUILD_DIR/wubu_ssm_chunked.o" \
    "$BUILD_DIR/wubu_poincare_gqa.o" \
    "$BUILD_DIR/wubu_poincare_gqa_backward.o" \
    "$BUILD_DIR/wubu_poincare_ssm_backward.o" \
    "$BUILD_DIR/wubu_nested_ssm.o" \
    "$BUILD_DIR/wubu_nested_ssm_backward.o" \
    "$BUILD_DIR/wubu_moe.o" \
    "$BUILD_DIR/wubu_tst.o" \
    "$BUILD_DIR/wubu_mobius_linear.o" \
    "$BUILD_DIR/thread_pool.o" \
    "$BUILD_DIR/qlearner.o" \
    "$BUILD_DIR/dequant_iq2_xxs.o" \
    "$BUILD_DIR/rsgd.o" \
    "$BUILD_DIR/quantized_matmul.o" \
    "$BUILD_DIR/cuda_kernels.o" \
    "$BUILD_DIR/flash_attn_q4_0_opt.o" \
    "$BUILD_DIR/flash_attn_q4_0_prefill_opt.o" \
    "$BUILD_DIR/wubu_model_gpu.o" \
    "$BUILD_DIR/rotorquant.o" \
    "$BUILD_DIR/poincare_attn.o" \
    "$BUILD_DIR/kv_arena.o" \
    "$BUILD_DIR/gpu_output_proj.o" \
    "$BUILD_DIR/gpu_quant_matmul.o" \
    "$BUILD_DIR/gpu_quant_matmul_row_major.o" \
    "$BUILD_DIR/gpu_moe_kernel.o" \
    "$BUILD_DIR/gpu_gemma4.o" \
    "$BUILD_DIR/gpu_gemma4_forward.o" \
    "$BUILD_DIR/gpu_ssm_recurrence.o" \
    $LDFLAGS $LIBM -lm

echo "Build complete: $BUILD_DIR/bench_512k_full"

echo "Build complete: $BUILD_DIR/bench_512k_full"
echo ""
echo "Usage: $BUILD_DIR/bench_512k_full <model_path> [ctx_len] [trials] [bench_type]"
echo "  bench_type: 0=NIAH 1=RULER 2=LongCode 3=AgentLong 4=LongBench-v2 5=MIR 6=ALL"