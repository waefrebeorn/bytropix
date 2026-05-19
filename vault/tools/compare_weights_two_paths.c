/**
 * compare_weights_two_paths.c
 * Load expert 64's gate weights via TWO paths and compare.
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);

#define D_MODEL 2048
#define D_FF 512
#define N_EXPERTS 256
#define EXPERT 64
#define IQ2_BLOCK 66

int main() {
    printf("=== Per-Expert Weight Compare ===\n\n");
    
    // Path A: gguf_read_tensor_f32
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) return 1;
    
    int64_t n_all = t->dims[0] * t->dims[1] * t->dims[2];
    float *path_a = (float *)malloc(n_all * sizeof(float));
    int ret = gguf_read_tensor_f32(ctx, t, path_a, n_all);
    printf("Path A (gguf_reader): %d elems\n", ret);
    
    int64_t e_off = (int64_t)EXPERT * D_MODEL * D_FF;
    printf("  E%d[0..3]: %.8f %.8f %.8f %.8f\n", EXPERT,
           path_a[e_off], path_a[e_off+1], path_a[e_off+2], path_a[e_off+3]);
    
    // Path B: direct fseek using gguf_reader offsets
    uint64_t abs_off = ctx->data_blob_offset + t->data_offset;
    printf("\nPath B (direct fseek @ %lu):\n", (unsigned long)abs_off);
    
    int bpe = D_FF * (D_MODEL / 256) * IQ2_BLOCK; // blocks per expert in bytes
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) return 1;
    fseek(f, abs_off + e_off / 256 * IQ2_BLOCK, SEEK_SET);
    // Wait: e_off is element offset. Block offset = e_off/256 * 66
    // Actually: expert offset in blocks = EXPERT * D_FF * (D_MODEL/256) = 64 * 512 * 8 = 262144 blocks
    int64_t block_off = (int64_t)EXPERT * D_FF * (D_MODEL / 256) * IQ2_BLOCK;
    fseek(f, abs_off + block_off, SEEK_SET);
    
    int64_t raw_n = D_MODEL * D_FF / 256 * IQ2_BLOCK;
    uint8_t *raw = (uint8_t *)malloc(raw_n);
    size_t nr = fread(raw, 1, raw_n, f);
    printf("  read %zu/%ld bytes\n", nr, (long)raw_n);
    if (nr != (size_t)raw_n) { printf("READ FAIL\n"); free(raw); fclose(f); goto end; }
    fclose(f);
    
    int e_elems = D_MODEL * D_FF;
    float *path_b = (float *)malloc(e_elems * sizeof(float));
    dequantize_iq2_xxs_row(raw, path_b, e_elems);
    printf("  E%d[0..3]: %.8f %.8f %.8f %.8f\n", EXPERT,
           path_b[0], path_b[1], path_b[2], path_b[3]);
    
    // Compare
    printf("\n=== COMPARE ===\n");
    double max_d = 0; int bad = 0, fi = -1;
    for (int i = 0; i < e_elems; i++) {
        double d = fabs(path_a[e_off + i] - path_b[i]);
        if (d > max_d) { max_d = d; fi = i; }
        if (d > 1e-6) bad++;
    }
    double dot=0,na=0,nb=0;
    for (int i = 0; i < e_elems; i++) {
        dot += path_a[e_off + i] * path_b[i];
        na += path_a[e_off + i] * path_a[e_off + i];
        nb += path_b[i] * path_b[i];
    }
    printf("  max_diff=%.10f (idx %d)\n", max_d, fi);
    printf("  mismatches(%%) = %d/%.0f (%.2f%%)\n", bad, (double)e_elems, 100.0*bad/e_elems);
    printf("  cos_sim=%.10f\n", dot/(sqrt(na)*sqrt(nb)+1e-30));
    printf("\n--- %s ---\n", max_d < 1e-6 ? "MATCH" : "DIFFER");

end:
    free(path_a); free(path_b); free(raw);
    gguf_close(ctx);
    return 0;
}
