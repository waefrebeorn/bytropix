#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    
    for (int i = 0; i < ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        if (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps") ||
            strstr(name, "ffn_gate_inp") || strstr(name, "ffn_gate_shexp") || strstr(name, "ffn_up_shexp") || strstr(name, "ffn_down_shexp")) {
            int64_t ne = ctx->tensors[i].n_dims > 0 ? ctx->tensors[i].dims[0] : 1;
            for (int d = 1; d < ctx->tensors[i].n_dims; d++)
                ne *= ctx->tensors[i].dims[d];
            printf("%-55s type=%d nd=%d dims=[%lld,%lld,%lld,%lld] ne=%lld\n", 
                   name, ctx->tensors[i].ggml_type, ctx->tensors[i].n_dims,
                   ctx->tensors[i].dims[0], ctx->tensors[i].dims[1], ctx->tensors[i].dims[2], ctx->tensors[i].dims[3],
                   ne);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
