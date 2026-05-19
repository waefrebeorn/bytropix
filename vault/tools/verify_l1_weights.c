/**
 * verify_l1_weights.c — Check that L1 MoE gate weights match between
 * our gguf_reader and direct file read
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
#define EXPERT 0
#define IQ2_BLOCK 66

int main() {
    printf("=== L1 Expert 0 Gate Weight Verify ===\n");
    
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.1.ffn_gate_exps.weight");
    if (!t) return 1;
    
    int64_t n_all = t->dims[0] * t->dims[1] * t->dims[2];
    float *path_a = (float *)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t, path_a, n_all);
    
    int64_t e_off = (int64_t)EXPERT * D_MODEL * D_FF;
    printf("L1 E0 gate[0..3]: %.8f %.8f %.8f %.8f\n",
           path_a[e_off], path_a[e_off+1], path_a[e_off+2], path_a[e_off+3]);
    
    // Compare L0 E0 vs L1 E0
    gguf_tensor_info *t0 = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    float *l0_w = (float *)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t0, l0_w, n_all);
    
    double dot=0, na=0, nb=0;
    for (int i = 0; i < D_MODEL * D_FF; i++) {
        float a = path_a[e_off + i];  // L1 E0
        float b = l0_w[e_off + i];    // L0 E0
        dot += (double)a * b; 
        na += (double)a * a; 
        nb += (double)b * b;
    }
    printf("L0 E0 vs L1 E0 gate: cos=%.10f\n", dot/(sqrt(na)*sqrt(nb)+1e-30));
    
    // Direct read for L1
    uint64_t abs_off = ctx->data_blob_offset + t->data_offset;
    int64_t block_off = (int64_t)EXPERT * D_FF * (D_MODEL / 256) * IQ2_BLOCK;
    
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    fseek(f, abs_off + block_off, SEEK_SET);
    int64_t raw_n = D_MODEL * D_FF / 256 * IQ2_BLOCK;
    uint8_t *raw = (uint8_t*)malloc(raw_n);
    size_t nr = fread(raw, 1, raw_n, f);
    fclose(f);
    
    int e_elems = D_MODEL * D_FF;
    float *path_b = (float*)malloc(e_elems * sizeof(float));
    dequantize_iq2_xxs_row(raw, path_b, e_elems);
    
    double max_d = 0;
    for (int i = 0; i < e_elems; i++) {
        double d = fabs(path_a[e_off + i] - path_b[i]);
        if (d > max_d) max_d = d;
    }
    printf("L1 E0 path_a vs path_b: max_diff=%.10f\n", max_d);
    printf("--- %s ---\n", max_d < 1e-6 ? "MATCH" : "DIFFER");
    
    free(path_a); free(l0_w); free(raw); free(path_b);
    gguf_close(ctx);
    return 0;
}
