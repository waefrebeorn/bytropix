#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = "/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf";
    if (argc > 1) path = argv[1];

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }

    for (int i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        if (strstr(t->name, ".0.") && (strstr(t->name, "attn") || strstr(t->name, "norm") || strstr(t->name, "output"))) {
            printf("[%3d] %-60s dims=[", i, t->name);
            for (int d = 0; d < t->n_dims; d++) {
                printf("%lld%s", (long long)t->dims[d], d < t->n_dims-1 ? "," : "");
            }
            printf("] type=%d offset=%llu\n", t->ggml_type, (unsigned long long)t->data_offset);
        }
    }

    gguf_close(ctx);
    return 0;
}
