#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    int n = ctx->n_tensors;

    // Find first IQ2_S tensor
    gguf_tensor_info *target = NULL;
    for (int i = 0; i < n; i++) {
        if (ctx->tensors[i].ggml_type == 18) {
            target = &ctx->tensors[i];
            break;
        }
    }
    if (!target) { printf("No IQ2_S tensors found\n"); return 1; }

    int64_t ne = 1;
    for (int d = 0; d < target->n_dims; d++) ne *= target->dims[d];
    printf("Dequantizing %s (ne=%ld, dims=%ld,%ld)...\n", target->name, (long)ne,
           (long)target->dims[0], (long)target->dims[1]);

    float *buf = (float *)malloc(ne * sizeof(float));
    int ok = gguf_read_tensor_f32(ctx, target, buf, ne);
    if (!ok) { printf("Failed to dequantize\n"); return 1; }

    // Full stats
    double sum = 0, sq_sum = 0;
    float min_val = buf[0], max_val = buf[0];
    int nan_count = 0, inf_count = 0;
    int extreme_count = 0;
    float threshold = 100.0f;

    for (int64_t j = 0; j < ne; j++) {
        float v = buf[j];
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
        sq_sum += v * v;
        if (fabs(v) > threshold) extreme_count++;
    }
    int64_t valid = ne - nan_count - inf_count;
    double mean = sum / valid;
    double variance = sq_sum / valid - mean * mean;
    double stddev = sqrt(variance);

    printf("\n=== Tensor Stats ===\n");
    printf("Min:          %f\n", min_val);
    printf("Max:          %f\n", max_val);
    printf("Mean:         %f\n", mean);
    printf("StdDev:       %f\n", stddev);
    printf("NaN count:    %d\n", nan_count);
    printf("Inf count:    %d\n", inf_count);
    printf("Extreme (|v|>100): %d / %ld (%.2f%%)\n",
           extreme_count, valid, 100.0 * extreme_count / valid);

    // Histogram
    printf("\n=== Histogram (log bins) ===\n");
    double bins[] = {0, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e10};
    int nbins = sizeof(bins)/sizeof(bins[0]);
    int counts[20] = {0};
    for (int64_t j = 0; j < valid; j++) {
        float v = fabs(buf[j]);
        for (int b = nbins-2; b >= 0; b--) {
            if (v >= bins[b]) { counts[b]++; break; }
        }
    }
    for (int b = 0; b < nbins-1; b++) {
        printf("  [%.0e, %.0e): %d\n", bins[b], bins[b+1], counts[b]);
    }

    // Find the actual extreme values
    printf("\n=== Top 10 maximum absolute values ===\n");
    // Just scan for the extremes
    float top_vals[10] = {0};
    int64_t top_idxs[10] = {0};
    for (int64_t j = 0; j < valid; j++) {
        float av = fabs(buf[j]);
        for (int k = 0; k < 10; k++) {
            if (av > top_vals[k]) {
                // shift down
                for (int kk = 9; kk > k; kk--) {
                    top_vals[kk] = top_vals[kk-1];
                    top_idxs[kk] = top_idxs[kk-1];
                }
                top_vals[k] = av;
                top_idxs[k] = j;
                break;
            }
        }
    }
    for (int k = 0; k < 10 && top_vals[k] > 0; k++) {
        printf("  [%d] idx=%ld val=%f\n", k, (long)top_idxs[k], buf[top_idxs[k]]);
    }

    free(buf);
    gguf_close(ctx);
    return 0;
}
