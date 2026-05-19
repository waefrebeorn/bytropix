#include "gguf_reader.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("fail\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Get quantized pointers for expert 0 gate
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    int gate_type = t->ggml_type;
    int64_t gate_bytes = gguf_raw_size(gate_type, (int64_t)2048 * 512);
    const uint8_t *gate_q = blob + t->data_offset;  // expert 0
    
    // Load embedding
    float *emb = (float *)malloc(2048 * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { printf("no emb\n"); return 1; }
    fseek(f, 248044LL * 2048 * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), 2048, f);
    fclose(f);
    
    // Simple RMS norm
    double sum_sq = 0;
    for (int i = 0; i < 2048; i++) sum_sq += (double)emb[i] * emb[i];
    double r = 1.0 / sqrt(sum_sq / 2048 + 1e-6);
    for (int i = 0; i < 2048; i++) emb[i] = (float)(emb[i] * r);
    
    // Test all 512 columns with direct iq2_xxs_dot_block calls (not moe_expert_forward_dequant)
    printf("Direct iq2_xxs_dot_block test:\n");
    float gate_out[512];
    const int blocks_per_col = 2048 / 256;
    const int gate_col_bytes = blocks_per_col * 66;
    for (int j = 0; j < 512; j++) {
        const uint8_t *qcol = gate_q + j * gate_col_bytes;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_col; b++)
            sum += iq2_xxs_dot_block(qcol + b * 66, emb + b * 256);
        gate_out[j] = sum;
        if (j < 5 || (j >= 507 && j < 512))
            printf("  gate[%d] = %.10f\n", j, gate_out[j]);
    }
    
    // Check for NaNs
    int nan_count = 0, inf_count = 0, huge_count = 0;
    for (int j = 0; j < 512; j++) {
        if (isnan(gate_out[j])) nan_count++;
        else if (isinf(gate_out[j])) inf_count++;
        else if (fabs(gate_out[j]) > 100.0f) huge_count++;
    }
    printf("NaNs: %d, Infs: %d, |val|>100: %d\n", nan_count, inf_count, huge_count);
    
    // Now test moe_expert_forward_dequant
    printf("\nmoe_expert_forward_dequant test:\n");
    t = gguf_find_tensor(ctx, "blk.0.ffn_up_exps.weight");
    const uint8_t *up_q = blob + t->data_offset;
    t = gguf_find_tensor(ctx, "blk.0.ffn_down_exps.weight");
    const uint8_t *down_q = blob + t->data_offset;
    
    float temp[512 * 3];
    float output[2048];
    moe_expert_forward_dequant(emb, gate_q, up_q, down_q, temp, output);
    
    for (int j = 0; j < 5; j++)
        printf("  out[%d] = %.10f\n", j, output[j]);
    
    nan_count = 0; inf_count = 0; huge_count = 0;
    for (int j = 0; j < 2048; j++) {
        if (isnan(output[j])) nan_count++;
        else if (isinf(output[j])) inf_count++;
        else if (fabs(output[j]) > 100.0f) huge_count++;
    }
    printf("output NaNs: %d, Infs: %d, |val|>100: %d\n", nan_count, inf_count, huge_count);
    
    free(emb);
    gguf_close(ctx);
    return 0;
}
