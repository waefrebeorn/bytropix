#ifndef WUBU_GPU_QUANT_MATMUL_H
#define WUBU_GPU_QUANT_MATMUL_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t wubu_cuda_quant_matmul_scratch(int n_rows, int n_cols, int quant_type);

int wubu_cuda_quant_matmul(const float *x, const uint8_t *W_q, int quant_type,
    int n_rows, int n_cols,
    float *y, float *scratch, size_t scratch_size,
    cudaStream_t stream);

// Row-major aware quant matmul (correct for GGUF layout)
// Each thread handles one input row. Supports Q5_K and Q6_K.
// Returns 1 on success, 0 on unknown quant type.
int wubu_cuda_quant_matmul_row_major(const float *x, const uint8_t *W_q, int quant_type,
    int n_rows, int n_cols,
    float *y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GPU_QUANT_MATMUL_H */
