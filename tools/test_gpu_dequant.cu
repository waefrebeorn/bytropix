/* test_gpu_dequant.c — Verify GPU Q4_K dequant matches CPU dequant.
 * Compile: nvcc -O3 -I include -gencode arch=compute_89,code=sm_89 tools/test_gpu_dequant.cu src/gpu_gemma4.o -o test_gpu_dequant -lcublas -lcudart
 */

#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define Q4K_BLOCK_SIZE 144
#define Q4K_N_ELEMS 256

extern __global__ void dequant_q4k_kernel(const uint8_t *W_q, float *W_f32, int64_t total_blocks);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    /* Load model just to get a Q4_K weight tensor */
    g4_model_t model;
    if (!g4_model_init(&model, argv[1])) { fprintf(stderr, "Failed to load\n"); return 1; }

    /* Pick a Q4_K weight tensor — ffn_gate of layer 0 is always Q4_K */
    g4_layer_t *l = &model.layers[0];
    g4_qweight_t *w = &l->ffn_gate;
    
    printf("ffn_gate: type=%d, raw_bytes=%lld, n_elems=%lld\n",
           w->ggml_type, (long long)w->raw_bytes, (long long)w->n_elems);

    if (w->ggml_type != GGML_TYPE_Q4_K) {
        printf("Not Q4_K (type=%d). Finding a Q4_K tensor...\n", w->ggml_type);
        for (int i = 0; i < model.n_layers; i++) {
            g4_layer_t *l2 = &model.layers[i];
            if (l2->ffn_gate.ggml_type == GGML_TYPE_Q4_K) {
                w = &l2->ffn_gate;
                l = l2;
                printf("  Found: layer %d ffn_gate (type=%d)\n", i, w->ggml_type);
                break;
            }
            if (l2->attn_q.ggml_type == GGML_TYPE_Q4_K) {
                w = &l2->attn_q;
                l = l2;
                printf("  Found: layer %d attn_q (type=%d)\n", i, w->ggml_type);
                break;
            }
        }
    }

    /* Get dimensions */
    int K = w->n_elems / (w->raw_bytes / Q4K_BLOCK_SIZE * Q4K_N_ELEMS);
    int N = 0;
    /* Infer N: n_elems / K, but K depends on layout */
    /* For ffn_gate: raw_bytes = 144 * (K*N) / 256, so K*N = n_elems */
    int64_t total_elems = w->n_elems;
    printf("  total_elems=%lld\n", (long long)total_elems);
    
    /* Test first 10 Q4_K blocks (2560 elements) */
    int test_blocks = 10;
    int test_elems = test_blocks * Q4K_N_ELEMS;
    printf("Dequant testing %d blocks (%d elements)...\n", test_blocks, test_elems);

    /* CPU dequant */
    float *cpu_out = (float*)malloc(test_elems * sizeof(float));
    for (int b = 0; b < test_blocks; b++) {
        const uint8_t *block = w->data + b * Q4K_BLOCK_SIZE;
        /* CPU reference: use the existing dequantize_q4_K_row */
        gguf_dequantize(w->data, w->ggml_type, test_elems, cpu_out);
    }

    /* GPU dequant */
    uint8_t *d_W;
    float *d_out, *h_out;
    cudaMalloc(&d_W, test_blocks * Q4K_BLOCK_SIZE);
    cudaMalloc(&d_out, test_elems * sizeof(float));
    h_out = (float*)malloc(test_elems * sizeof(float));
    
    cudaMemcpy(d_W, w->data, test_blocks * Q4K_BLOCK_SIZE, cudaMemcpyHostToDevice);
    dequant_q4k_kernel<<<test_blocks, Q4K_N_ELEMS>>>(d_W, d_out, test_blocks);
    cudaMemcpy(h_out, d_out, test_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Compare */
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < test_elems; i++) {
        float diff = fabsf(cpu_out[i] - h_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            if (errors < 10)
                printf("  MISMATCH [%d]: CPU=%f GPU=%f diff=%f\n", i, cpu_out[i], h_out[i], diff);
            errors++;
        }
    }

    printf("Result: %d/%d errors, max_diff=%f\n", errors, test_elems, max_diff);
    if (errors == 0 && max_diff < 0.001f) {
        printf("✅ Q4_K GPU dequant CORRECT\n");
    } else {
        printf("❌ Q4_K GPU dequant INCORRECT\n");
    }

    cudaFree(d_W); cudaFree(d_out);
    free(cpu_out); free(h_out);
    g4_model_destroy(&model);
    return errors > 0 ? 1 : 0;
}
