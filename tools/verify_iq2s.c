#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    const char *path = argv[1];
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    
    // Find first IQ2_S tensor
    int iq2s_count = 0;
    gguf_tensor_info *iq2s_tensor = NULL;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].ggml_type == GGML_TYPE_IQ2_S) {
            iq2s_count++;
            if (!iq2s_tensor) iq2s_tensor = &ctx->tensors[i];
        }
    }
    printf("Total tensors: %d, IQ2_S tensors: %d\n", ctx->n_tensors, iq2s_count);
    
    if (!iq2s_tensor) { fprintf(stderr, "No IQ2_S tensor found\n"); gguf_close(ctx); return 1; }
    
    // Get dimensions
    int64_t n_elems = 1;
    for (int d = 0; d < iq2s_tensor->n_dims; d++) n_elems *= iq2s_tensor->dims[d];
    printf("First IQ2_S tensor: %s, type=%d, shape=[", iq2s_tensor->name, iq2s_tensor->ggml_type);
    for (int d = 0; d < iq2s_tensor->n_dims; d++) printf("%s%ld", d ? "," : "", iq2s_tensor->dims[d]);
    printf("], n_elems=%ld\n", n_elems);
    
    // Dequantize (max 2.1M elements — ~8MB, fine for 0.8B model)
    int64_t max_elems = 2100000;
    float *buf = (float *)malloc(max_elems * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed\n"); gguf_close(ctx); return 1; }
    
    int n_read = gguf_read_tensor_f32(ctx, iq2s_tensor, buf, max_elems);
    printf("Dequantized %d elements\n", n_read);
    
    if (n_read > 0) {
        // Statistics
        double sum = 0, sum2 = 0;
        float vmin = buf[0], vmax = buf[0];
        int extreme_count = 0;
        for (int i = 0; i < n_read; i++) {
            float v = buf[i];
            sum += v;
            sum2 += v * v;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
            if (fabs(v) > 10.0f) extreme_count++;
        }
        double mean = sum / n_read;
        double variance = sum2 / n_read - mean * mean;
        double stddev = sqrt(variance > 0 ? variance : 0);
        
        printf("First 8 values: ");
        for (int i = 0; i < 8 && i < n_read; i++) printf("%+.6f ", buf[i]);
        printf("\n");
        printf("Min: %.3f, Max: %.3f, Mean: %.3f, StdDev: %.3f\n", vmin, vmax, mean, stddev);
        printf("|v|>10 count: %d / %d (%.1f%%)\n", extreme_count, n_read, 100.0 * extreme_count / n_read);
        
        if (stddev < 2.0f) {
            printf("VERDICT: PASS — IQ2_S dequant producing O(0.1-1) range\n");
        } else {
            printf("VERDICT: FAIL — still producing extreme values\n");
        }
    }
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
