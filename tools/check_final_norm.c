#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not open %s\n", path);
        return 1;
    }
    
    // Buffer the data blob for fast access
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "WARN: gguf_buffer_data failed, trying gguf_read_tensor_f32 anyway\n");
    }
    
    // Find output_norm.weight
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output_norm.weight");
    if (!t) {
        fprintf(stderr, "FAIL: output_norm.weight not found\n");
        gguf_close(ctx);
        return 1;
    }
    
    // Print tensor info
    printf("=== output_norm.weight ===\n");
    printf("  n_dims=%d, dims=[", t->n_dims);
    for (int d = 0; d < t->n_dims; d++) {
        if (d > 0) printf(",");
        printf("%ld", t->dims[d]);
    }
    printf("]\n");
    printf("  ggml_type=%d, data_offset=%lu\n", t->ggml_type, (unsigned long)t->data_offset);
    
    // Calculate total elements
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    printf("  n_elems=%ld\n", (long)n_elems);
    
    // Allocate buffer and read
    float *buf = (float*)malloc(n_elems * sizeof(float));
    if (!buf) {
        fprintf(stderr, "FAIL: malloc failed\n");
        gguf_close(ctx);
        return 1;
    }
    
    int got = gguf_read_tensor_f32(ctx, t, buf, n_elems);
    if (got <= 0) {
        fprintf(stderr, "FAIL: gguf_read_tensor_f32 returned %d\n", got);
        free(buf);
        gguf_close(ctx);
        return 1;
    }
    printf("  got=%d values\n", got);
    
    // Compute statistics
    float min_val = INFINITY;
    float max_val = -INFINITY;
    double sum = 0.0;
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    int one_count = 0;
    
    int limit = (n_elems < 32) ? (int)n_elems : 32;
    printf("\n  First %d values:\n  ", limit);
    for (int i = 0; i < got && i < n_elems; i++) {
        float v = buf[i];
        if (i < limit) {
            printf("%.10f%s", (double)v, (i < limit-1) ? ", " : "\n");
        }
        
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += (double)v;
        if (v == 0.0f) zero_count++;
        if (v == 1.0f) one_count++;
    }
    
    int valid = got - nan_count - inf_count;
    double mean = (valid > 0) ? sum / valid : 0.0;
    
    printf("\n  === Statistics ===\n");
    printf("  min    = %.10f\n", (double)min_val);
    printf("  max    = %.10f\n", (double)max_val);
    printf("  mean   = %.10f\n", mean);
    printf("  NaNs   = %d\n", nan_count);
    printf("  Infs   = %d\n", inf_count);
    printf("  zeros  = %d\n", zero_count);
    printf("  ones   = %d\n", one_count);
    
    // Check for suspicious patterns
    printf("\n  === Suspicious Pattern Check ===\n");
    if (nan_count > 0)  printf("  WARNING: contains %d NaN values!\n", nan_count);
    if (inf_count > 0)  printf("  WARNING: contains %d Inf values!\n", inf_count);
    if (zero_count == got) printf("  WARNING: ALL values are zero!\n");
    if (one_count == got)  printf("  WARNING: ALL values are one!\n");
    
    // Check if all same value
    float first = buf[0];
    int all_same = 1;
    for (int i = 1; i < got && i < n_elems; i++) {
        if (buf[i] != first) { all_same = 0; break; }
    }
    if (all_same) printf("  WARNING: ALL %d values are identical (%.10f)!\n", got, (double)first);
    
    if (nan_count == 0 && inf_count == 0 && zero_count < got && one_count < got && !all_same)
        printf("  OK: No suspicious patterns detected.\n");
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
