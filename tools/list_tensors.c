#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        printf("%s\n", ctx->tensors[i].name);
    }
    gguf_close(ctx);
    return 0;
}
