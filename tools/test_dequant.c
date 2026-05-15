/**
 * test_dequant: Dump first 32 floats from a specific tensor
 * to compare against llama.cpp reference.
 * Usage: ./test_dequant [tensor_name]
 * Default: blk.0.attn_qkv.weight
 */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *tname = argc > 2 ? argv[2] : "blk.0.attn_qkv.weight";
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, tname);
    if (!t) { fprintf(stderr, "Tensor '%s' not found\n", tname); gguf_close(ctx); return 1; }
    
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    
    printf("Tensor: %s dims=[", tname);
    for (int d = 0; d < t->n_dims; d++) printf("%s%ld", d?",":"", t->dims[d]);
    printf("] type=%d n_elems=%ld\n", t->ggml_type, n_elems);
    
    // Read first 256 elements (QK_K = 256, one dequant block)
    int chunk = n_elems < 256 ? n_elems : 256;
    float *buf = (float *)malloc(chunk * sizeof(float));
    int ret = gguf_read_tensor_f32(ctx, t, buf, chunk);
    printf("Read %d elements\n", ret);
    
    // Print first 32
    for (int i = 0; i < 32; i++) {
        printf("  [%d] = %f\n", i, buf[i]);
    }
    
    // Also print last 32 of first block
    if (chunk >= 256) {
        printf("  ... middle of block ...\n");
        for (int i = 224; i < 256; i++) {
            printf("  [%d] = %f\n", i, buf[i]);
        }
    }
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
