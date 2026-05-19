#ifndef WUBU_GPU_QUANT_MATMUL_H
#define WUBU_GPU_QUANT_MATMUL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <cuda_runtime.h>
extern "C" {
#endif

// ================================================================
// Quantized GPU matmul — keep weights on GPU in native quant format
// ================================================================

// Query scratch size for a quantized GPU matmul of given dimensions
// Returns 0 if the quant type isn't supported on GPU yet.
// For non-quant types (F32, etc.), returns 0 — caller should use cuBLAS.
size_t wubu_cuda_quant_matmul_scratch(int n_rows, int n_cols, int quant_type);

// Quantized matmul: y [n_cols] = x [n_rows] @ W_q [n_rows, n_cols]^T (conceptually)
// x: [n_rows] F32 input (device)
// W_q: [bytes_total] quantized weight data, column-major (device)
// quant_type: GGML_TYPE_Q5_K or GGML_TYPE_Q6_K (others return 0)
// n_rows: input dimension (D_MODEL or VALUE_DIM)
// n_cols: output dimension (CONV_DIM, VALUE_DIM, or D_MODEL)
// y: [n_cols] F32 output (device)
// scratch: optional scratch buffer (if non-NULL, use instead of internal alloc)
// scratch_size: size of scratch buffer (from query_scratch)
// stream: CUDA stream
// Returns 1 on success, 0 on failure (unsupported type, etc.)
int wubu_cuda_quant_matmul(const float *x, const uint8_t *W_q, int quant_type,
    int n_rows, int n_cols,
    float *y, float *scratch, size_t scratch_size,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
