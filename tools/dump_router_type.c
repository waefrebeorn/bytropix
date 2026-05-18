#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "Failed\n"); return 1; }
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        if (strstr(t->name, "gate_inp") || strstr(t->name, "shexp")) {
            int64_t ne = 1;
            for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
            printf("%-50s type=%2d dims=%d [", t->name, t->ggml_type, t->n_dims);
            for (int d = 0; d < t->n_dims; d++) printf("%s%ld", d?",":"", (long)t->dims[d]);
            printf("] n_elems=%ld\n", (long)ne);
        }
    }
    gguf_close(ctx);
    return 0;
}
