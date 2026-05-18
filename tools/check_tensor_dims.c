#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    gguf_buffer_data(ctx);

    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (!t) { fprintf(stderr, "output.weight not found\n"); return 1; }
    printf("output.weight: dims=[%ld,%ld,%ld,%ld] ndims=%d type=%d\n",
           (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], (long)t->dims[3],
           t->n_dims, t->ggml_type);

    // Also check token_embd.weight
    t = gguf_find_tensor(ctx, "token_embd.weight");
    if (t) {
        printf("token_embd.weight: dims=[%ld,%ld,%ld,%ld] ndims=%d type=%d\n",
               (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], (long)t->dims[3],
               t->n_dims, t->ggml_type);
    }

    // Check what other tensors look like
    t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (t) {
        printf("blk.0.attn_q.weight: dims=[%ld,%ld,%ld,%ld]\n",
               (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], (long)t->dims[3]);
    }

    gguf_close(ctx);
    return 0;
}
