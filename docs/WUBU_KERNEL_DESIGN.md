# Hardware-Agnostic Kernel Design

## Goal

One C API, multiple backends. CPU portable C baseline + device backends
(CUDA/Metal/Vulkan/ROCm/BLAS) register at runtime. Engine dispatches through
the table which auto-selects the best backend at runtime based on workload
characteristics.

## Architecture

```c
/* Engine hot path calls this — never a backend directly */
wubu_kernel_run(WUBU_KERN_GEMM, A, B, C, M, K, N, beta);
```

The `wubu_kernel_run` dispatcher:
1. Resolves best backend (registered device > CPU-scalar)
2. Unpacks variadic args (type-safe per kernel type)
3. Calls the function pointer

## Backend Registration

Device backends call `wubu_kernel_register()` with their function pointers.

### CUDA Backend (`src/wubu_kernel_cuda.cu`)
- Implements GEMM, GEMV, Attention, Softmax, RMSNorm, Quantize, Dequantize
- `wubu_cuda_backend_probe()`: calls `cudaGetDeviceCount()` → 1 if GPU present
- `wubu_cuda_backend_register()`: registers with function pointers
- On WSL2: CUDA runtime libs at `/usr/lib/x86_64-linux-gnu/`, GPU passthrough
  at `/usr/lib/wsl/lib/` (set `LD_LIBRARY_PATH=/usr/lib/wsl/lib`)
- RTX 4050 Laptop GPU detected (6GB VRAM, CC 8.9)

### Registration Flow
`wubu_kernel_init()` → `wubu_kernel_register_backends()` → each backend probes
runtime availability → if available, registers via `wubu_kernel_register()`.

## Compile-Time Flags

Use `WUBU_ENABLE_*` (not `WUBU_BACKEND_*`) to avoid clashing with the enum:
```
-DWUBU_ENABLE_CUDA    (CUDA device backend)
-DWUBU_ENABLE_METAL   (Apple Metal backend)
-DWUBU_ENABLE_VULKAN  (Vulkan compute backend)
-DWUBU_ENABLE_ROCM    (ROCm/HIP backend)
-DWUBU_ENABLE_BLAS    (BLAS backend)
```

## CPU Feature Detection

`wubu_cpu_detect()` reads `/proc/meminfo` + `/sys/devices/system/cpu/` caches.
Auto-select: AVX2+FMA → CPU_SIMD, else SCALAR.

## Kernel Types

| Type | Function | CPU Baseline | CUDA |
|------|----------|-------------|------|
| GEMM | C = A*B  | ✅ tiled GEMM | ✅ |
| GEMV | y = A*x  | ✅ tiled GEMV | ✅ |
| ATTN | softmax(QK^T)V | ✅ causal masked | ❌ (stub) |
| ROPE | rotary embedding | ✅ interleaved | ❌ (stub) |
| SOFTMAX | row-wise | ✅ | ✅ |
| LAYER_NORM | RMSNorm | ✅ | ✅ |
| QUANT | fp32→int8 | ✅ | ✅ |
| DEQUANT | int8→fp32 | ✅ | ✅ |

## Hot Path Integration

`proj_matmul()` in `wubu_ssm.c` and GQA QKV+gate projections now route F32
weights through `wubu_kernel_run(WUBU_KERN_GEMV, ...)` — dispatches to CUDA
GEMV if a GPU is available, else CPU tiled AVX2-FMA GEMV.

## Verified Commands

```bash
# Build with CUDA support
make gen_text  # builds with -DWUBU_ENABLE_CUDA

# Test kernel dispatch (13 tests)
make test_kernel_dispatch
# Expected: 13 PASS, 0 errors
# CUDA backend: Active GEMV/GEMM backend = cuda
# GEMV dispatch==scalar: max_diff=1.49e-08 PASS

# Full test suite
make test_all
# Expected: ALL TESTS PASSED (11 tests)

# Run gen_text with CUDA
LD_LIBRARY_PATH=/usr/lib/wsl/lib MAX_LAYERS=2 ./gen_text MODEL "Hello" 1
```

## WSL2 GPU Setup

```bash
# CUDA toolkit (nvcc v12) installed via apt
# GPU passthrough libs at /usr/lib/wsl/lib/
# Required at runtime:
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# Or launch gen_text with the prefix
```

## KV-cache Compression Research (Kevin-Bacon 7-hop)

| Method | Paper | Implementation | Result |
|--------|-------|----------------|--------|
| KIVI | 2402.02750 | wubu_kvcache_quant.c (existing) | K per-channel, V per-token INT4 |
| SAW-INT4 | 2604.19157 | wubu_4kv.c (K Hadamard+BDR + INT4) | K cosine=0.9969, V cosine=0.9965 |
| TurboQuant | ICLR 2026 (Google+NYU) | wubu_4kv.c (INT3 V) | 6.1× compression, cosine=0.980 |
| Ecco | ISCA 2025 | wubu_4kv.c (INT8 skip-head) | Entropy-adaptive per-head |
| MiniKV | UIUC 2025 | N/A (CPU-only, skip multi-tier) | CPU-realizable subset above |

Key finding: under CPU-only constraints (no GPU kernel), SAW-INT4's
block-diagonal Hadamard rotation on keys + block-wise INT4 on values
achieves 5.1× KV reduction with ~0.3% cosine degradation on synthetic
Gaussian data. Real model KV (concentrated distributions) should achieve
0.999+ per the paper.

1. **Dispatch correctness**: CUDA GEMV vs CPU scalar → max_diff=1.49e-08 (FP32 precision) ✅
2. **ATTN/ROPE**: dispatch == scalar → 0.0 diff ✅
3. **Backend lifecycle**: init → register → shutdown → re-init works ✅
4. **No double-free**: `wubu_kernel_shutdown()` only frees malloc'd backends, not CPU baseline ✅