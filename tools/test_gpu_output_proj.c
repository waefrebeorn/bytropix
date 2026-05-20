/**
 * test_gpu_output_proj.c — Standalone test: compare CPU vs GPU output proj.
 * Loads model, runs forward to get hidden states, computes output proj
 * on both CPU and GPU, compares results.
 *
 * Build: make test_gpu_output_proj
 * Usage: GPU_VERBOSE=1 OMP_NUM_THREADS=16 ./test_gpu_output_proj
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include "gpu_output_proj.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double wall_clock(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    mdl.enable_moe = true;

    // Init GPU output proj from model weights
    if (!gpu_output_init(mdl.output_weight_q, D_MODEL, mdl.vocab_size, mdl.output_weight_type)) {
        fprintf(stderr, "GPU init failed\n");
        wubu_model_free(&mdl);
        return 1;
    }

    int D = D_MODEL, V = mdl.vocab_size;

    // Create a test hidden state (first row of token_embd)
    float x[D];
    if (mdl.token_embd) {
        memcpy(x, mdl.token_embd, D * sizeof(float));
    } else {
        for (int i = 0; i < D; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // CPU output proj via quantized_matmul (same path as gen_text)
    float cpu_logits[V];
    double t0 = wall_clock();
    quantized_matmul(x, mdl.output_weight_q, mdl.output_weight_type,
                     D, V, 0, cpu_logits);
    double t_cpu = (wall_clock() - t0) * 1000.0;

    // GPU output proj
    float gpu_logits[V];
    t0 = wall_clock();
    if (!gpu_output_project(x, gpu_logits)) {
        fprintf(stderr, "GPU project failed\n");
        gpu_output_cleanup();
        wubu_model_free(&mdl);
        return 1;
    }
    double t_gpu = (wall_clock() - t0) * 1000.0;

    // Compare
    printf("=== GPU vs CPU Output Projection ===\n\n");
    printf("Matrix: [%d] @ [%d, %d] = [%d]\n", D, D, V, V);
    printf("CPU: %.3f ms  GPU: %.3f ms\n\n", t_cpu, t_gpu);

    // Compare logits
    double cos_dot = 0, cos_n1 = 0, cos_n2 = 0;
    double max_err = 0; int max_i = -1;
    int cpu_max = 0, gpu_max = 0;
    float cpu_mv = cpu_logits[0], gpu_mv = gpu_logits[0];

    for (int i = 0; i < V; i++) {
        cos_dot += (double)cpu_logits[i] * (double)gpu_logits[i];
        cos_n1  += (double)cpu_logits[i] * (double)cpu_logits[i];
        cos_n2  += (double)gpu_logits[i] * (double)gpu_logits[i];
        double e = fabs((double)cpu_logits[i] - (double)gpu_logits[i]);
        if (e > max_err) { max_err = e; max_i = i; }
        if (cpu_logits[i] > cpu_mv) { cpu_mv = cpu_logits[i]; cpu_max = i; }
        if (gpu_logits[i] > gpu_mv) { gpu_mv = gpu_logits[i]; gpu_max = i; }
    }

    double cos_sim = cos_dot / (sqrt(cos_n1) * sqrt(cos_n2));
    printf("Cosine similarity: %.10f\n", cos_sim);
    printf("Max element error: %.10f at index %d\n", max_err, max_i);
    printf("  CPU[%d]=%.6f  GPU[%d]=%.6f\n", max_i, cpu_logits[max_i], max_i, gpu_logits[max_i]);
    printf("Argmax: CPU=%d (%.4f)  GPU=%d (%.4f)\n", cpu_max, cpu_mv, gpu_max, gpu_mv);

    printf("\nFirst 10 logits:\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] CPU=%.6f GPU=%.6f diff=%.10f\n", i, cpu_logits[i], gpu_logits[i],
               (double)(cpu_logits[i] - gpu_logits[i]));

    printf("\nLast 10 logits:\n");
    for (int i = V-10; i < V; i++)
        printf("  [%d] CPU=%.6f GPU=%.6f\n", i, cpu_logits[i], gpu_logits[i]);

    gpu_output_cleanup();
    wubu_model_free(&mdl);
    return 0;
}
