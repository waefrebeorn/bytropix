# Hardware-Agnostic Kernel Design

## Goal

One C API, multiple backends. CPU portable C baseline + device backends
(CUDA/Metal/Vulkan/ROCm/BLAS) register at init. Engine dispatches through the
table which auto-selects the best backend at runtime based on workload
characteristics.

## Architecture

```c
/* Engine hot path calls this — never a backend directly */
wubu_kernel_run(WUBU_KERN_GEMM, A, B, C, M, K, N, beta);
```

The `wubu_kernel_run` dispatcher:
1. Resolves best backend (registered device > CPU-simd > CPU-scalar)
2. Unpacks variadic args (type-safe per kernel type)
3. Calls the function pointer

## Backend Registration

Device backends call `wubu_kernel_register()` with their function pointers:
```c
wubu_kernel_backend_t cuda_backend = {
    .id = WUBU_BACKEND_CUDA,
    .name = "cuda",
    .gemm = cuda_gemm,
    .gemv = cuda_gemv,
    .attn = cuda_attention,
    .supports = cuda_supports,
    .next = NULL
};
wubu_kernel_register(WUBU_BACKEND_CUDA, "cuda", &cuda_backend);
```

## CPU Feature Detection

`wubu_cpu_detect()` reads `/proc/meminfo` + `/sys/devices/system/cpu/` caches.
Auto-select: AVX2+FMA → CPU_SIMD, else SCALAR.

## Kernel Types

| Type | Function | CPU Baseline |
|------|----------|-------------|
| GEMM | C = A*B  | ✅ tiled GEMM |
| GEMV | y = A*x  | ✅ tiled GEMV |
| ATTN | softmax(QK^T)V | ❌ (gap) |
| ROPE | rotary embedding | ❌ (gap) |
| SOFTMAX | row-wise | ✅ |
| LAYER_NORM | RMSNorm | ✅ |
| QUANT | fp32→int8 | ✅ |
| DEQUANT | int8→fp32 | ✅ |

## Integration Plan

Phase 1: CPU-SIMD baseline — wire `wubu_kernel_run` calls into `quantized_matmul.c`
hot path (GEMV dispatch already done).

Phase 2: ATTN + ROPE CPU baselines — implement attention and ROPE in the
scalar/CPU_SIMD slot so the dispatch table is complete.

Phase 3: Device backends — stub registration points for CUDA/Vulkan/Metal
(compile-time `#if WUBU_HAS_*` guards).

## WSL2 Substrate

- 6 P-cores, AVX2+FMA present
- No GPU (CUDA/Metal/Vulkan backends are `#if 0` stubs)
- ~13GB RAM — arena scratch allocators, no bulk checkpoint reads