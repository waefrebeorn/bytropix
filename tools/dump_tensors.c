#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "FAIL\n"); return 1; }

    printf("=== Layer 0 tensors ===\n");
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        if (strstr(name, "blk.0.")) {
            printf("  %s: dims=[", name);
            for (int d = 0; d < ctx->tensors[i].n_dims; d++) {
                if (d > 0) printf(",");
                printf("%ld", (long)ctx->tensors[i].dims[d]);
            }
            printf("] type=%d\n", ctx->tensors[i].ggml_type);
        }
    }

    // Check if any tensor name contains "ssm_d" or "d_scale"
    printf("\n=== Searching for ssm_d / d_scale / skip connection tensors ===\n");
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        if (strstr(name, "ssm_d") || strstr(name, "d_scale") || 
            strstr(name, "dt_bias") || strstr(name, "ssm_a") || strstr(name, "ssm_beta")) {
            printf("  %s: dims=[", name);
            for (int d = 0; d < ctx->tensors[i].n_dims; d++) {
                if (d > 0) printf(",");
                printf("%ld", (long)ctx->tensors[i].dims[d]);
            }
            printf("] type=%d\n", ctx->tensors[i].ggml_type);
        }
    }

    gguf_close(ctx);
    return 0;
}
