#ifndef GPU_OUTPUT_PROJ_H
#define GPU_OUTPUT_PROJ_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GPU output projection
// Dequants Q4_K weight → F32, uploads to GPU, creates cuBLAS handle
bool gpu_output_init(const uint8_t *weight_q, int D, int V, int weight_type);

// Run GPU output projection (SGEMM via cuBLAS)
bool gpu_output_project(const float *input, float *output);

// Cleanup GPU resources
void gpu_output_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
