#!/usr/bin/env bash
# check_cuda_env.sh -- verify + REPAIR the wubuwizard CUDA layout on this
# WSL2 box. The recurring break: /usr/local/cuda-13.1 is a PARTIAL .run
# install (nvcc only) and its include/ + lib64/ must be symlinked to the
# apt toolkit, or every CUDA build silently degrades to CPU.
#
# Usage:  tools/check_cuda_env.sh          # check only
#         sudo tools/check_cuda_env.sh     # check + repair
set -u

CUDA_ROOT=/usr/local/cuda-13.1
APT_INC=/usr/include/cuda
APT_LIB=/usr/lib/x86_64-linux-gnu
WSL_LIB=/usr/lib/wsl/lib

echo "== CUDA environment check =="
fail=0

# 1. the GPU itself (Windows driver passthrough)
if [ -c /dev/dxg ] || [ -e "$WSL_LIB/libcuda.so.1" ]; then
    echo "  [ok] GPU passthrough ($WSL_LIB/libcuda.so.1)"
else
    echo "  [FAIL] no WSL GPU passthrough -- install the Windows NVIDIA driver"; fail=1
fi
if command -v nvidia-smi.exe >/dev/null 2>&1; then
    nvidia-smi.exe --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/  [gpu] /'
fi

# 2. nvcc
if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
    echo "  [ok] nvcc $CUDA_ROOT/bin/nvcc"
else
    echo "  [FAIL] nvcc missing at $CUDA_ROOT/bin/nvcc"; fail=1
fi

# 3. include/
if [ -d "$CUDA_ROOT/include" ]; then
    echo "  [ok] include -> $(readlink -f "$CUDA_ROOT/include")"
else
    echo "  [FAIL] $CUDA_ROOT/include missing"
    [ "${1:-}" = "repair" ] || [ "$(id -u)" = "0" ] && ln -s "$APT_INC" "$CUDA_ROOT/include" && echo "  [repair] ln -s $APT_INC $CUDA_ROOT/include"
    [ -d "$CUDA_ROOT/include" ] || fail=1
fi

# 4. lib64/
if [ -f "$CUDA_ROOT/lib64/libcublas.so.12" ] || [ -f "$CUDA_ROOT/lib64/libcublas.so" ]; then
    echo "  [ok] lib64 -> $(readlink -f "$CUDA_ROOT/lib64")"
else
    echo "  [FAIL] $CUDA_ROOT/lib64 missing libcublas"
    [ "${1:-}" = "repair" ] || [ "$(id -u)" = "0" ] && ln -s "$APT_LIB" "$CUDA_ROOT/lib64" && echo "  [repair] ln -s $APT_LIB $CUDA_ROOT/lib64"
    [ -f "$CUDA_ROOT/lib64/libcublas.so.12" ] || [ -f "$CUDA_ROOT/lib64/libcublas.so" ] || fail=1
fi

# 5. /usr/local/cuda symlink
if [ "$(readlink -f /usr/local/cuda)" = "$(readlink -f "$CUDA_ROOT")" ]; then
    echo "  [ok] /usr/local/cuda -> $CUDA_ROOT"
else
    echo "  [FAIL] /usr/local/cuda -> $(readlink -f /usr/local/cuda 2>/dev/null || echo missing)"
    [ "${1:-}" = "repair" ] || [ "$(id -u)" = "0" ] && rm -f /usr/local/cuda && ln -s "$CUDA_ROOT" /usr/local/cuda && echo "  [repair] /usr/local/cuda -> $CUDA_ROOT"
    [ "$(readlink -f /usr/local/cuda)" = "$(readlink -f "$CUDA_ROOT")" ] || fail=1
fi

if [ "$fail" = "0" ]; then
    echo "== CUDA OK. Run: make cuda_check && make test_gpu_matmul =="
    exit 0
else
    echo "== CUDA BROKEN. Run: sudo tools/check_cuda_env.sh repair, then re-run. =="
    exit 1
fi
