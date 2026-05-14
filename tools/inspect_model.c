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
    printf("Total tensors: %d\n\n", n);

    int iq2s_count = 0;
    for (int i = 0; i < n; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        int64_t ne = 1;
        for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        printf("  [%3d] %-50s dims=", i, t->name);
        for (int d = 0; d < t->n_dims; d++) printf("%ld%c", (long)t->dims[d], d+1<t->n_dims?',':' ');
        printf("n_dims=%d type=%d ne=%ld\n",
               t->n_dims, t->ggml_type, (long)ne);
        if (t->ggml_type == 18) { // IQ2_S
            printf("         ^^^ IQ2_S tensor!\n");
            iq2s_count++;
        }
    }
    printf("\nTotal IQ2_S tensors: %d\n", iq2s_count);

    // If there are IQ2_S tensors, try to dequant one and check range
    if (iq2s_count > 0) {
        for (int i = 0; i < n; i++) {
            gguf_tensor_info *t = &ctx->tensors[i];
            if (t->ggml_type != 18) continue;
            int64_t ne = 1;
            for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
            printf("\nDequantizing %s (nelements=%ld)...\n", t->name, (long)ne);
            float *buf = (float *)malloc(ne * sizeof(float));
            if (!buf) { printf("  malloc failed!\n"); break; }
            int ok = gguf_read_tensor_f32(ctx, t, buf, ne);
            if (ok) {
                float min = buf[0], max = buf[0], sum = 0;
                int nan_count = 0, inf_count = 0;
                int64_t checked = ne > 1024*1024 ? 1024*1024 : ne;
                for (int64_t j = 0; j < checked; j++) {
                    float v = buf[j];
                    if (isnan(v)) nan_count++;
                    else if (isinf(v)) inf_count++;
                    else {
                        if (v < min) min = v;
                        if (v > max) max = v;
                        sum += v;
                    }
                }
                int64_t valid = checked - nan_count - inf_count;
                printf("  First 8 values: ");
                for (int j = 0; j < 8 && j < ne; j++) printf("%+.6f ", buf[j]);
                printf("\n  Range: [%f, %f], mean=%f, nan=%d, inf=%d\n",
                       min, max, valid > 0 ? sum/valid : 0.0f, nan_count, inf_count);
            } else {
                printf("  Failed to dequantize!\n");
            }
            free(buf);
            break; // just first IQ2_S tensor
        }
    }

    gguf_close(ctx);
    return 0;
}
