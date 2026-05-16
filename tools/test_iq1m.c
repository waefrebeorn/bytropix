#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    // Find IQ1_M tensors (layers 34, 38, 39)
    for (int l = 34; l <= 39; l++) {
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (!t) { printf("  %s: NOT FOUND\n", name); continue; }
        printf("Tensor: %s  type=%d  dims=[%ld,%ld,%ld]\n", 
               name, t->ggml_type, (long)t->dims[0], (long)t->dims[1], (long)t->dims[2]);
        
        if (t->ggml_type != GGML_TYPE_IQ1_M) {
            printf("  SKIP (not IQ1_M, type=%d)\n", t->ggml_type);
            continue;
        }
        
        // Read raw data
        gguf_buffer_data(ctx);
        int64_t n_elems = t->dims[0] * t->dims[1] * t->dims[2];
        float *buf = (float *)malloc(n_elems * sizeof(float));
        if (!gguf_read_tensor_f32(ctx, t, buf, n_elems)) {
            printf("  FAILED to read\n");
            free(buf);
            continue;
        }
        
        // Stats
        float sum = 0, sum2 = 0, vmin = 1e30, vmax = -1e30;
        for (int64_t i = 0; i < n_elems; i++) {
            sum += buf[i];
            sum2 += buf[i] * buf[i];
            if (buf[i] < vmin) vmin = buf[i];
            if (buf[i] > vmax) vmax = buf[i];
        }
        float mean = sum / n_elems;
        float std = sqrtf(sum2 / n_elems - mean * mean);
        
        printf("  elems=%ld  mean=%f  std=%f  min=%f  max=%f\n", 
               (long)n_elems, mean, std, vmin, vmax);
        printf("  first 10: ");
        for (int i = 0; i < 10 && i < n_elems; i++)
            printf("%.6f ", buf[i]);
        printf("\n");
        
        free(buf);
    }
    
    gguf_close(ctx);
    printf("DONE\n");
    return 0;
}
