#include "gguf_reader.h"
#include <stdio.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "Failed to open model\n"); return 1; }

    int counts[256] = {0};
    printf("All tensors with type code:\n");
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        counts[t->ggml_type]++;
        // Print MoE tensors and down_exps specifically
        if (strstr(t->name, "ffn_gate_exps") || strstr(t->name, "ffn_up_exps") || 
            strstr(t->name, "ffn_down_exps") || strstr(t->name, "output.weight") ||
            strstr(t->name, "token_embd")) {
            int64_t n_elems = 1;
            for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
            printf("  %-45s type=%3d dims=[", t->name, t->ggml_type);
            for (int d = 0; d < t->n_dims; d++) printf("%s%ld", d?",":"", (long)t->dims[d]);
            printf("] n_elems=%ld\n", (long)n_elems);
        }
    }

    printf("\nType count summary:\n");
    for (int i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            printf("  type %3d: %d tensors\n", i, counts[i]);
        }
    }

    // Also check specific layers for down_exps variation
    printf("\nDown experts across layers 0, 33, 34, 37, 38, 39:\n");
    for (int l = 0; l <= 39; l++) {
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) printf("  %s type=%d\n", name, t->ggml_type);
    }

    gguf_close(ctx);
    return 0;
}
