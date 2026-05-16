#include "gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    
    for (int l = 0; l < 40; l++) {
        char nm[256];
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, nm);
        if (t) {
            printf("L%d ffn_gate_inp: dims=%d", l, t->n_dims);
            for (int i = 0; i < t->n_dims; i++) printf(" %ld", t->dims[i]);
            printf(" type=%d\n", t->ggml_type);
        }
        
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", l);
        t = gguf_find_tensor(ctx, nm);
        if (t) {
            printf("L%d ffn_gate_exps: dims=%d", l, t->n_dims);
            for (int i = 0; i < t->n_dims; i++) printf(" %ld", t->dims[i]);
            printf(" type=%d\n", t->ggml_type);
        }
        
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", l);
        t = gguf_find_tensor(ctx, nm);
        if (t) {
            printf("L%d ffn_gate_shexp: dims=%d", l, t->n_dims);
            for (int i = 0; i < t->n_dims; i++) printf(" %ld", t->dims[i]);
            printf(" type=%d\n", t->ggml_type);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
