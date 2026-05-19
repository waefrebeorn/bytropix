#ifndef GPU_OUTPUT_PROJ_H
#define GPU_OUTPUT_PROJ_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GPU output projection
// Dequants Q4_K weight → F32, uploads to GPU, creates cuBLAS handle
// GPU_BATCH env var controls max batch size for batched prefill (default 1)
bool gpu_output_init(const uint8_t *weight_q, int D, int V, int weight_type);

// Batched output projection for prefill: T hidden states → T logit vectors
// input: [T, D_MODEL] contiguous row-major, output: [T, vocab_size]
// T must be <= GPU_BATCH value set at init time
bool gpu_output_project_batch(const float *input, float *output, int T);

// Run single-token GPU output projection (SGEMM via cuBLAS)
bool gpu_output_project(const float *input, float *output);

// Cleanup GPU resources
void gpu_output_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
