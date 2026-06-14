#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (!t) { fprintf(stderr, "not found\n"); return 1; }
    printf("attn_q: type=%d dims=%lld %lld\n", t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);
    printf("t->data_offset = %llu\n", t->data_offset);
    
    // Total elements
    int64_t total_elems = t->dims[0] * t->dims[1];
    printf("Total elements = %ld\n", total_elems);
    int64_t blocks = (total_elems + 255) / 256;
    printf("Blocks = %ld\n", blocks);
    printf("Block size 144 -> %ld bytes\n", blocks * 144);
    printf("Block size 176 -> %ld bytes\n", blocks * 176);
    int64_t raw_sz = gguf_raw_size(t->ggml_type, total_elems);
    printf("gguf_raw_size = %ld bytes\n", raw_sz);
    
    // Check second block starts where
    const uint8_t *wq = blob + t->data_offset;
    printf("\nSecond block start (offset 144):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] = %02x\n", i, wq[144+i]);
    }
    printf("\nSecond block start (offset 176):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] = %02x\n", i, wq[176+i]);
    }
    
    // Also check what tensor type 2 means
    printf("\nGGML type 2 = Q4_K\n");
    // Check if there's quantization version in metadata
    // Look for gguf tensor info
    
    gguf_close(ctx);
    return 0;
}
