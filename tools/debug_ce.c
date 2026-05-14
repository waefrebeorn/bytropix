/**
 * debug_ce.c — Diagnose why CE loss is 6.6e10
 * Checks: output.weight Q4_K dequant range, a few vocab logits
 */
#include "gguf_reader.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Load the output.weight tensor
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (!t) { fprintf(stderr, "No output.weight\n"); gguf_close(ctx); return 1; }
    
    printf("output.weight: dims=[%ld,%ld] type=%d n_elems=%ld\n",
           (long)t->dims[0], (long)t->dims[1],
           t->ggml_type, (long)(t->dims[0]*t->dims[1]));
    
    int D = t->dims[0];
    int V = t->dims[1];
    int64_t n_elems = (int64_t)D * V;
    
    float *weight = (float *)malloc(n_elems * sizeof(float));
    int nread = gguf_read_tensor_f32(ctx, t, weight, n_elems);
    printf("Read %d elements\n", nread);
    
    // Check a few rows
    for (int row = 0; row < 5 && row < V; row++) {
        float mn = 1e30, mx = -1e30;
        for (int k = 0; k < D; k++) {
            float v = weight[row * D + k];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        printf("  Row %d/%d: range [%.4e, %.4e]\n", row, V, mn, mx);
    }
    
    // Check a few random rows
    int rows[] = {100, 1000, 10000, 100000, 200000, V/2, V-1};
    for (int ri = 0; ri < 7; ri++) {
        int row = rows[ri];
        if (row >= V) continue;
        float mn = 1e30, mx = -1e30;
        for (int k = 0; k < D; k++) {
            float v = weight[row * D + k];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        printf("  Row %d/%d: range [%.4e, %.4e]\n", row, V, mn, mx);
    }
    
    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0;
    for (int64_t i = 0; i < n_elems && i < 10000000; i++) {
        if (isnan(weight[i])) nan_count++;
        if (isinf(weight[i])) inf_count++;
    }
    printf("  NaN count (first 10M): %d, Inf count: %d\n", nan_count, inf_count);
    
    // Now simulate one token's vocab logits
    float h[2048];
    for (int k = 0; k < D; k++) h[k] = -3.0f + (float)rand() / RAND_MAX * 5.0f - 250.0f * (float)(k % 2);
    
    double max_l = -1e30;
    double target_logit = 0.0;
    for (int j = 0; j < V && j < 1000; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)h[k] * (double)weight[j * D + k];
        if (j == 0) target_logit = sum;
        if (sum > max_l) max_l = sum;
    }
    printf("\nFirst 1000 vocab logits: max_l=%.4f, target_logit=%.4f\n", max_l, target_logit);
    
    // Do full 248K just for the first j=100..110
    double logits_100_110[10];
    for (int j = 100; j < 110 && j < V; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)h[k] * (double)weight[j * D + k];
        logits_100_110[j-100] = sum;
    }
    printf("  Logits[100:110]:");
    for (int i = 0; i < 10; i++) printf(" %.2f", logits_100_110[i]);
    printf("\n");
    
    free(weight);
    gguf_close(ctx);
    return 0;
}
