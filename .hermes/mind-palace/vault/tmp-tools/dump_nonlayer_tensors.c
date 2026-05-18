#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        // Only print tensors NOT in blk.0-blk.39 range (i.e. non-layer tensors)
        if (strncmp(t->name, "blk.", 4) != 0) {
            int64_t n = 1;
            for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
            printf("%-45s dims=[", t->name);
            for (int d = 0; d < t->n_dims; d++) printf("%s%ld", d?",":"", (long)t->dims[d]);
            printf("] type=%d n_elems=%ld\n", t->ggml_type, (long)n);
        }
    }
    printf("--- end ---\n");
    gguf_close(ctx);
    return 0;
}
