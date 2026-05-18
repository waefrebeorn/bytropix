// Minimal test: struct layout from ggml-common.h, functions from libggml-cpu.so
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

// Must initialize ggml to populate lookup tables
extern struct ggml_context * ggml_init(struct ggml_init_params params);
void ggml_backend_cpu_init(void);  // from libggml-cpu.so

int main() {
    // Initialize ggml
    struct ggml_init_params params = {.mem_size = 1024*1024, .mem_buffer = NULL, .no_alloc = true};
    struct ggml_context *ctx = ggml_init(params);
    ggml_backend_cpu_init();
    
    printf("sizeof(block_q4_K) = %zu\n", sizeof(block_q4_K));
    printf("sizeof(block_q5_K) = %zu\n", sizeof(block_q5_K));
    printf("sizeof(block_q8_K) = %zu\n", sizeof(block_q8_K));
    
    // Create Q5_K block using CORRECT struct
    block_q5_K q5;
    memset(&q5, 0, sizeof(q5));
    q5.d = ggml_fp32_to_fp16(1.0f);
    q5.dmin = 0;
    memset(q5.scales, 0xFF, K_SCALE_SIZE);
    memset(q5.qh, 0xFF, QK_K/8);
    memset(q5.qs, 0xFF, QK_K/2);
    
    // Create Q8_K quantized version of all-1 input
    float x[QK_K];
    for (int i = 0; i < QK_K; i++) x[i] = 1.0f;
    
    block_q8_K q8;
    quantize_row_q8_K(x, &q8, QK_K);
    
    printf("Q8_K: d=%f qs[0]=%d\n", q8.d, q8.qs[0]);
    
    // Call vec_dot (from libggml-cpu.so)
    float result = 0.0f;
    ggml_vec_dot_q5_K_q8_K(QK_K, &result, 0, &q5, 0, &q8, 0, 1);
    printf("vec_dot_q5_K_q8_K = %f\n", (double)result);
    
    // Q4_K
    block_q4_K q4;
    memset(&q4, 0, sizeof(q4));
    q4.d = ggml_fp32_to_fp16(1.0f);
    q4.dmin = 0;
    memset(q4.scales, 0xFF, K_SCALE_SIZE);
    memset(q4.qs, 0xFF, QK_K/2);
    
    float r4 = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &r4, 0, &q4, 0, &q8, 0, 1);
    printf("vec_dot_q4_K_q8_K = %f\n", (double)r4);
    
    return 0;
}
