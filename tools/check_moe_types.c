#include "../include/gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", argv[1]); return 1; }

    // Check first layer MoE types
    for (int layer = 0; layer < 3; layer++) {
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        int gate_t = t ? t->ggml_type : -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
        t = gguf_find_tensor(ctx, name);
        int up_t = t ? t->ggml_type : -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
        t = gguf_find_tensor(ctx, name);
        int down_t = t ? t->ggml_type : -1;
        
        int64_t gate_sz = gguf_raw_size(gate_t, (int64_t)2048 * 512);
        int64_t up_sz = gguf_raw_size(up_t, (int64_t)2048 * 512);
        int64_t down_sz = gguf_raw_size(down_t, (int64_t)512 * 2048);
        
        printf("Layer %d: gate=%d (%ldB/exp) up=%d (%ldB/exp) down=%d (%ldB/exp)\n",
               layer, gate_t, (long)gate_sz, up_t, (long)up_sz, down_t, (long)down_sz);
    }

    gguf_close(ctx);
    return 0;
}
