#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    printf("Total tensors: %ld\n", (long)ctx->n_tensors);
    
    // Count types
    int type_counts[256] = {0};
    for (int i = 0; i < ctx->n_tensors && i < 1000; i++) {
        int t = ctx->tensors[i].ggml_type;
        if (t >= 0 && t < 256) type_counts[t]++;
    }
    
    printf("\nTensor types:\n");
    for (int t = 0; t < 256; t++) {
        if (type_counts[t] > 0) {
            printf("  type %d: %d tensors\n", t, type_counts[t]);
        }
    }
    
    // Look for blk.0.ffn_gate_exps.weight
    printf("\nLooking for target tensor...\n");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (t) {
        printf("Found: %s\n", t->name);
        printf("  n_dims=%d, dims=[%ld,%ld,%ld,%ld]\n", t->n_dims, 
               (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], (long)t->dims[3]);
        printf("  ggml_type=%d\n", t->ggml_type);
        printf("  data_offset=%lu\n", (unsigned long)t->data_offset);
        
        int64_t n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        printf("  total_elements=%ld\n", (long)n_elems);
        
        int64_t raw_size = gguf_raw_size(t->ggml_type, n_elems);
        printf("  raw_size=%ld\n", (long)raw_size);
    } else {
        printf("Not found. Listing first 5 tensors with 'exps':\n");
        for (int i = 0; i < ctx->n_tensors; i++) {
            if (strstr(ctx->tensors[i].name, "exps")) {
                printf("  [%d] %s type=%d dims=[%ld,%ld]\n", i, ctx->tensors[i].name,
                       ctx->tensors[i].ggml_type,
                       (long)ctx->tensors[i].dims[0], (long)ctx->tensors[i].dims[1]);
                if (strstr(ctx->tensors[i].name, "blk.0")) break;
            }
        }
    }

    gguf_close(ctx);
    return 0;
}
