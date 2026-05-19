#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// iq2_xxs_dot_block prototype
extern float iq2_xxs_dot_block(const uint8_t *q, const float *x);

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("fail\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Get gate_exps
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    const uint8_t *gate_all = blob + t->data_offset;
    
    // Load embedding
    float *emb = (float *)malloc(2048 * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { printf("no emb\n"); return 1; }
    fseek(f, 248044LL * 2048 * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), 2048, f);
    fclose(f);
    
    // RMS norm
    double sum_sq = 0;
    for (int i = 0; i < 2048; i++) sum_sq += (double)emb[i] * emb[i];
    double r = 1.0 / sqrt(sum_sq / 2048 + 1e-6);
    for (int i = 0; i < 2048; i++) emb[i] = (float)(emb[i] * r);
    
    // Test first block of expert 0
    // Block 0 covers elements 0-255
    const uint8_t *block0 = gate_all;  // expert 0, first 256 elements
    float dot0 = iq2_xxs_dot_block(block0, emb);
    printf("Block 0 dot: %.10f\n", dot0);
    
    // Compare against F32: load same expert's F32 data
    t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    int64_t expert_elems = (int64_t)2048 * 512;
    float *f32_gate = (float *)malloc(expert_elems * sizeof(float));
    // Read only expert 0
    // gguf_read_tensor_f32 doesn't support offset, so read expert 0 only
    // by reading the full tensor and discarding the rest
    // Actually let's allocate and read just one expert
    int64_t total_gate = expert_elems * 256; // all 256 experts
    float *f32_full = (float *)malloc(total_gate * sizeof(float));
    if (!f32_full) { printf("malloc failed\n"); return 1; }
    if (!gguf_read_tensor_f32(ctx, t, f32_full, total_gate)) {
        printf("read failed\n"); return 1;
    }
    // Copy expert 0
    memcpy(f32_gate, f32_full, expert_elems * sizeof(float));
    free(f32_full);
    
    // Compute F32 dot for first block
    double f32_dot0 = 0;
    for (int i = 0; i < 256; i++)
        f32_dot0 += (double)emb[i] * (double)f32_gate[i];  // expert 0, col 0
    
    printf("F32 block 0 dot: %.10f\n", f32_dot0);
    printf("Difference: %.10f\n", dot0 - f32_dot0);
    
    // Try column 5 (j=5): elements [5*2048 .. 5*2048+255]
    int col = 5;
    const uint8_t *block_col5 = gate_all + col * 8 * 66;  // gate_col_bytes = 528
    float dot_q5 = iq2_xxs_dot_block(block_col5, emb);
    
    double f32_dot5 = 0;
    for (int i = 0; i < 256; i++)
        f32_dot5 += (double)emb[i] * (double)f32_gate[col * 2048 + i];
    
    printf("\nColumn 5 quantized dot: %.10f\n", dot_q5);
    printf("Column 5 F32 dot: %.10f\n", f32_dot5);
    
    // Now test ALL 512 columns for expert 0
    printf("\nTesting all 512 columns for expert 0...\n");
    double max_diff = 0; int max_j = -1;
    int bad_count = 0;
    for (int j = 0; j < 512; j++) {
        const uint8_t *qcol = gate_all + j * 8 * 66;
        double q_dot = 0;
        for (int b = 0; b < 8; b++)
            q_dot += iq2_xxs_dot_block(qcol + b * 66, emb + b * 256);
        
        double f_dot = 0;
        for (int i = 0; i < 2048; i++)
            f_dot += (double)emb[i] * (double)f32_gate[j * 2048 + i];
        
        double d = fabs(q_dot - f_dot);
        if (d > max_diff) { max_diff = d; max_j = j; }
        if (d > 1.0f) bad_count++;
    }
    printf("Max diff: %.6f at col %d\n", max_diff, max_j);
    printf("Columns with diff > 1.0: %d / 512\n", bad_count);
    if (bad_count > 0 && max_j >= 0) {
        // Recheck bad column
        int j = max_j;
        const uint8_t *qcol = gate_all + j * 8 * 66;
        for (int b = 0; b < 8; b++) {
            double d = iq2_xxs_dot_block(qcol + b * 66, emb + b * 256);
            printf("  block %d: q_dot=%.6f\n", b, d);
        }
    }
    
    free(emb);
    free(f32_gate);
    gguf_close(ctx);
    return 0;
}
