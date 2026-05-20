#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *tensor_name = argc > 2 ? argv[2] : "output.weight";
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, tensor_name) == 0) {
            int64_t n_elems = 1;
            for (int d = 0; d < ctx->tensors[i].n_dims; d++)
                n_elems *= ctx->tensors[i].dims[d];
            printf("Tensor: %s\n", ctx->tensors[i].name);
            printf("  Type: %d (0=F32 1=F16 13=Q5_K 14=Q6_K 15=Q4_K 16=IQ2_XXS 18=IQ3_XXS)\n", ctx->tensors[i].ggml_type);
            printf("  Dims: ");
            for (int d = 0; d < ctx->tensors[i].n_dims; d++)
                printf("%lld ", (long long)ctx->tensors[i].dims[d]);
            printf("\n  Elems: %lld\n", (long long)n_elems);
            printf("  Offset: %llu\n", (unsigned long long)ctx->tensors[i].data_offset);
        }
    }
    
    const char *critical[] = {"token_embd.weight", "output_norm.weight", NULL};
    for (int c = 0; critical[c]; c++) {
        for (int i = 0; i < (int)ctx->n_tensors; i++) {
            if (strcmp(ctx->tensors[i].name, critical[c]) == 0) {
                int64_t n_elems = 1;
                for (int d = 0; d < ctx->tensors[i].n_dims; d++)
                    n_elems *= ctx->tensors[i].dims[d];
                printf("Tensor: %s  type=%d dims=", critical[c], ctx->tensors[i].ggml_type);
                for (int d = 0; d < ctx->tensors[i].n_dims; d++)
                    printf("%lld ", (long long)ctx->tensors[i].dims[d]);
                printf(" elems=%lld\n", (long long)n_elems);
            }
        }
    }
    
    gguf_close(ctx);
    return 0;
}
