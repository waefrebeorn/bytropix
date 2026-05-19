/**
 * dump_gate_expert0.c — Dump first column of gate weights for expert 0
 * from our loaded F32 buffer. Print raw hex bytes of first 8 blocks too.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) return 1;
    
    // Dump gate_exps first 32 floats for expert 0
    printf("=== Expert 0 gate_exps F32 (loaded) ===\n");
    printf("First 32 floats:\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%3d] = %.10f (0x%08x)\n", i, moe.ffn_gate_exps[i],
               *(uint32_t*)&moe.ffn_gate_exps[i]);
    }
    
    // Also dump the raw GGUF data at the gate_exps offset
    gguf_tensor_info *t = gguf_find_tensor(mdl.gguf_ctx, "blk.0.ffn_gate_exps.weight");
    if (t) {
        printf("\n=== Raw GGUF data at tensor offset ===\n");
        uint64_t pos = mdl.gguf_ctx->data_blob_offset + t->data_offset;
        // Use global model path
        FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
        if (f) {
            fseek(f, pos, SEEK_SET);
            uint8_t raw[32];
            fread(raw, 1, 32, f);
            printf("First 32 bytes at file pos %lu:\n", (unsigned long)pos);
            for (int i = 0; i < 32; i++) printf("%02x ", raw[i]);
            printf("\n");
            fclose(f);
        }
    }
    
    wubu_moe_free_layer(&moe);
    wubu_model_free(&mdl);
    return 0;
}
