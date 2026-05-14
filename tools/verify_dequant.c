#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;
    
    // Check IQ2_S (type 22) — find smallest tensor
    printf("=== IQ2_S (type 22) ===\n");
    int iq2s_count = 0;
    gguf_tensor_info *smallest_iq2s = NULL;
    int64_t smallest_elems = 1LL << 60;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].ggml_type == GGML_TYPE_IQ2_S) {
            iq2s_count++;
            int64_t n = 1;
            for (int d = 0; d < ctx->tensors[i].n_dims; d++) n *= ctx->tensors[i].dims[d];
            if (n < smallest_elems) { smallest_elems = n; smallest_iq2s = &ctx->tensors[i]; }
        }
    }
    printf("Count: %d\n", iq2s_count);
    
    if (smallest_iq2s) {
        printf("Smallest: %s (%ld elems)\n", smallest_iq2s->name, smallest_elems);
        float *buf = malloc(smallest_elems * sizeof(float));
        int n = gguf_read_tensor_f32(ctx, smallest_iq2s, buf, smallest_elems);
        printf("Dequantized %d elements\n", n);
        if (n > 0) {
            double sum = 0, sum2 = 0;
            float vmin = buf[0], vmax = buf[0];
            int extreme = 0;
            for (int i = 0; i < n; i++) {
                sum += buf[i]; sum2 += buf[i]*buf[i];
                if (buf[i] < vmin) vmin = buf[i];
                if (buf[i] > vmax) vmax = buf[i];
                if (fabs(buf[i]) > 10) extreme++;
            }
            printf("First 8: ");
            for (int i = 0; i < 8; i++) printf("%+.4f ", buf[i]);
            printf("\nRange: [%.3f, %.3f] mean=%.3f stddev=%.3f |v|>10=%d/%d\n",
                   vmin, vmax, sum/n, sqrt(sum2/n - (sum/n)*(sum/n)), extreme, n);
            printf("VERDICT: %s\n", extreme==0 ? "PASS" : "FAIL");
        }
        free(buf);
    }
    
    // Check IQ3_XXS (type 18)
    printf("\n=== IQ3_XXS (type 18) ===\n");
    int iq3_count = 0;
    gguf_tensor_info *smallest_iq3 = NULL;
    smallest_elems = 1LL << 60;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].ggml_type == GGML_TYPE_IQ3_XXS) {
            iq3_count++;
            int64_t n = 1;
            for (int d = 0; d < ctx->tensors[i].n_dims; d++) n *= ctx->tensors[i].dims[d];
            if (n < smallest_elems) { smallest_elems = n; smallest_iq3 = &ctx->tensors[i]; }
        }
    }
    printf("Count: %d\n", iq3_count);
    
    if (smallest_iq3) {
        printf("Smallest: %s (%ld elems)\n", smallest_iq3->name, smallest_elems);
        float *buf = malloc(smallest_elems * sizeof(float));
        int n = gguf_read_tensor_f32(ctx, smallest_iq3, buf, smallest_elems);
        printf("Dequantized %d elements\n", n);
        if (n > 0) {
            double sum = 0, sum2 = 0;
            float vmin = buf[0], vmax = buf[0];
            int extreme = 0;
            for (int i = 0; i < n; i++) {
                sum += buf[i]; sum2 += buf[i]*buf[i];
                if (buf[i] < vmin) vmin = buf[i];
                if (buf[i] > vmax) vmax = buf[i];
                if (fabs(buf[i]) > 10) extreme++;
            }
            printf("First 8: ");
            for (int i = 0; i < 8; i++) printf("%+.4f ", buf[i]);
            printf("\nRange: [%.3f, %.3f] mean=%.3f stddev=%.3f |v|>10=%d/%d\n",
                   vmin, vmax, sum/n, sqrt(sum2/n - (sum/n)*(sum/n)), extreme, n);
            printf("VERDICT: %s\n", extreme==0 ? "PASS" : "FAIL");
        }
        free(buf);
    }
    
    gguf_close(ctx);
    return 0;
}
