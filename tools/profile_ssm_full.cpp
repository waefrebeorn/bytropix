/**
 * Direct test of wubu_model_gpu_ssm_forward_full with PROFILE_SSM timing.
 * Bypasses wubu_model.c build issues. Links directly against CUDA objects.
 * Usage: ./profile_ssm_full model.gguf [layer=0] [n_tokens=8]
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Forward decls from wubu_model_gpu.cu
extern "C" int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);
extern "C" int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
    const float *h_norm, int C, float *h_attn_out);
extern "C" void wubu_model_gpu_free(wubu_model_t *model);
extern "C" int wubu_cuda_init(cublasHandle_t *handle, cudaStream_t *stream);

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    int n_tokens = argc > 3 ? atoi(argv[3]) : 8;

    printf("=== Profile: wubu_model_gpu_ssm_forward_full ===\n");
    printf("Model: %s  Layer: %d  N=%d\n\n", gguf_path, layer, n_tokens);

    // Load full model
    wubu_model_t mdl;
    memset(&mdl, 0, sizeof(mdl));
    if (!wubu_model_init(&mdl, gguf_path)) {
        fprintf(stderr, "FAIL: model init\n");
        return 1;
    }
    printf("Model loaded: %d layers\n", mdl.n_layers);
    if (layer >= mdl.n_layers) {
        fprintf(stderr, "Layer %d out of range (max %d)\n", layer, mdl.n_layers);
        return 1;
    }
    if (!mdl.layers[layer].is_ssm) {
        fprintf(stderr, "Layer %d is not SSM\n", layer);
        return 1;
    }

    // Init GPU
    if (!wubu_model_gpu_init(&mdl, 4096, 1024)) {
        fprintf(stderr, "FAIL: GPU init\n");
        return 1;
    }
    printf("GPU initialized\n");

    // Generate random input
    float *h_norm = (float*)malloc(n_tokens * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < n_tokens * D_MODEL; i++)
        h_norm[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    float *h_out = (float*)calloc(n_tokens * D_MODEL, sizeof(float));

    // Prefill (batched)
    printf("\n=== N=%d prefill (batched) ===\n", n_tokens);
    for (int w = 0; w < 2; w++) {
        wubu_model_gpu_ssm_forward_full(&mdl, layer, h_norm, n_tokens, h_out);
    }
    double t0 = wall_time();
    if (!wubu_model_gpu_ssm_forward_full(&mdl, layer, h_norm, n_tokens, h_out)) {
        fprintf(stderr, "FAIL\n");
        return 1;
    }
    double t1 = wall_time();
    printf("Total: %.3f ms (%.1f tok/s)\n", (t1-t0)*1000, n_tokens/(t1-t0));

    // Decode N=1 (multiple tokens)
    printf("\n=== N=1 decode x%d ===\n", n_tokens);
    // Warmup
    wubu_model_gpu_ssm_forward_full(&mdl, layer, h_norm, 1, h_out);
    t0 = wall_time();
    for (int t = 0; t < n_tokens; t++) {
        if (!wubu_model_gpu_ssm_forward_full(&mdl, layer, h_norm + t * D_MODEL, 1, h_out + t * D_MODEL)) {
            fprintf(stderr, "FAIL at token %d\n", t);
            return 1;
        }
    }
    t1 = wall_time();
    printf("Total: %.3f ms (%.1f tok/s)\n", (t1-t0)*1000, n_tokens/(t1-t0));

    free(h_norm);
    free(h_out);
    wubu_model_gpu_free(&mdl);
    wubu_model_free(&mdl);
    printf("\nDone.\n");
    return 0;
}
