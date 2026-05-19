#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// constants from wubu_moe.h
#define N_EXPERTS 256
#define N_ACTIVE_EXPTS 8
#define D_MODEL 2048
#define D_FF 512

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("fail\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    printf("Model loaded, blob at %p\n", (void*)blob);
    
    // Get quantized type and offsets
    char name[256];
    snprintf(name, sizeof(name), "blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    printf("ffn_gate_exps: type=%d offset=%llu dims=%llu %llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], (unsigned long long)t->dims[2]);
    
    snprintf(name, sizeof(name), "blk.0.ffn_up_exps.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_up_exps: type=%d offset=%llu dims=%llu %llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], (unsigned long long)t->dims[2]);
    
    snprintf(name, sizeof(name), "blk.0.ffn_down_exps.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_down_exps: type=%d offset=%llu dims=%llu %llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], (unsigned long long)t->dims[2]);
    
    snprintf(name, sizeof(name), "blk.0.ffn_gate_inp.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_gate_inp: type=%d offset=%llu dims=%llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
    
    // Load just ONE expert's F32 weights for expert 0
    // dequant ffn_gate_exps for expert 0 only
    snprintf(name, sizeof(name), "blk.0.ffn_gate_exps.weight");
    t = gguf_find_tensor(ctx, name);
    int64_t expert_elems = (int64_t)D_MODEL * D_FF;  // 2048 * 512 = 1,048,576
    float *gate_f32_expert0 = (float *)malloc(expert_elems * sizeof(float));
    
    // We need to read only expert 0's slice from the GGUF tensor
    // The tensor data is contiguous with expert as the outer dim
    // We dequant only the first expert's portion
    int64_t expert_offset = 0;  // expert 0 starts at offset 0
    int64_t raw_elem_offset = expert_offset * expert_elems;
    int64_t raw_byte_offset = raw_elem_offset; // but we need type-aware sizing
    
    // Read using gguf_read_tensor_f32 with offset
    // GGUF reader doesn't support offset reads, so we read the full tensor
    // and just take expert 0
    float *full_gate = (float *)malloc(expert_elems * N_EXPERTS * sizeof(float));
    if (gguf_read_tensor_f32(ctx, t, full_gate, expert_elems * N_EXPERTS)) {
        memcpy(gate_f32_expert0, full_gate, expert_elems * sizeof(float));
        printf("Expert 0 gate F32 loaded\n");
    } else {
        printf("F32 read failed\n");
        free(full_gate);
        return 1;
    }
    free(full_gate);
    
    // Same for up
    snprintf(name, sizeof(name), "blk.0.ffn_up_exps.weight");
    t = gguf_find_tensor(ctx, name);
    float *up_f32_expert0 = (float *)malloc(expert_elems * sizeof(float));
    float *full_up = (float *)malloc(expert_elems * N_EXPERTS * sizeof(float));
    if (gguf_read_tensor_f32(ctx, t, full_up, expert_elems * N_EXPERTS)) {
        memcpy(up_f32_expert0, full_up, expert_elems * sizeof(float));
        printf("Expert 0 up F32 loaded\n");
    }
    free(full_up);
    
    // Down: dims = [D_FF, D_MODEL] = [512, 2048] per expert
    snprintf(name, sizeof(name), "blk.0.ffn_down_exps.weight");
    t = gguf_find_tensor(ctx, name);
    int64_t down_elems = (int64_t)D_FF * D_MODEL;
    float *down_f32_expert0 = (float *)malloc(down_elems * sizeof(float));
    float *full_down = (float *)malloc(down_elems * N_EXPERTS * sizeof(float));
    if (gguf_read_tensor_f32(ctx, t, full_down, down_elems * N_EXPERTS)) {
        memcpy(down_f32_expert0, full_down, down_elems * sizeof(float));
        printf("Expert 0 down F32 loaded\n");
    }
    free(full_down);
    
    // Load embedding
    float *emb = (float *)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { printf("no emb file\n"); return 1; }
    fseek(f, 248044LL * D_MODEL * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), D_MODEL, f);
    fclose(f);
    
    // Apply RMS norm
    float *normed = (float *)malloc(D_MODEL * sizeof(float));
    // Use all-ones norm weight
    float *norm_w = (float *)malloc(D_MODEL * sizeof(float));
    for (int i = 0; i < D_MODEL; i++) norm_w[i] = 1.0f;
    
    // RMS norm: out[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps)
    double sum_sq = 0;
    for (int i = 0; i < D_MODEL; i++) sum_sq += (double)emb[i] * emb[i];
    double r = 1.0 / sqrt(sum_sq / D_MODEL + 1e-6);
    for (int i = 0; i < D_MODEL; i++) normed[i] = (float)(emb[i] * r);
    
    printf("\n--- Comparing F32 vs Quantized matmul for expert 0 ---\n");
    
    // ---- F32 SGEMM path ----
    float gate_f32[D_FF], up_f32[D_FF], act_f32[D_FF], out_f32[D_MODEL];
    
    for (int j = 0; j < D_FF; j++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)normed[k] * (double)gate_f32_expert0[k + j * D_MODEL];
        gate_f32[j] = (float)sum;
    }
    for (int j = 0; j < D_FF; j++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)normed[k] * (double)up_f32_expert0[k + j * D_MODEL];
        up_f32[j] = (float)sum;
    }
    for (int j = 0; j < D_FF; j++) {
        float g = gate_f32[j];
        float s = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        act_f32[j] = s * up_f32[j];
    }
    for (int j = 0; j < D_MODEL; j++) {
        double sum = 0;
        for (int k = 0; k < D_FF; k++)
            sum += (double)act_f32[k] * (double)down_f32_expert0[k + j * D_FF];
        out_f32[j] = (float)sum;
    }
    
    printf("F32 expert 0 output: first 5 values:\n");
    for (int j = 0; j < 5; j++) printf("  [%d] %.10f\n", j, out_f32[j]);
    
    // ---- Quantized matmul path ----
    // Get quantized pointers
    snprintf(name, sizeof(name), "blk.0.ffn_gate_exps.weight");
    t = gguf_find_tensor(ctx, name);
    int64_t gate_bytes = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    const uint8_t *gate_q = blob + t->data_offset;  // expert 0
    
    snprintf(name, sizeof(name), "blk.0.ffn_up_exps.weight");
    t = gguf_find_tensor(ctx, name);
    int64_t up_bytes = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    const uint8_t *up_q = blob + t->data_offset;
    
    snprintf(name, sizeof(name), "blk.0.ffn_down_exps.weight");
    t = gguf_find_tensor(ctx, name);
    int64_t down_bytes = gguf_raw_size(t->ggml_type, (int64_t)D_FF * D_MODEL);
    const uint8_t *down_q = blob + t->data_offset;
    
    printf("gate_bytes=%lld up_bytes=%lld down_bytes=%lld\n",
           (long long)gate_bytes, (long long)up_bytes, (long long)down_bytes);
    
    // Use the moe_expert_forward_dequant function
    extern void moe_expert_forward_dequant(const float *x,
        const uint8_t *gate_q, const uint8_t *up_q, const uint8_t *down_q,
        float *temp, float *output);
    
    float temp[D_FF * 3];
    float out_q[D_MODEL];
    moe_expert_forward_dequant(normed, gate_q, up_q, down_q, temp, out_q);
    
    printf("Quantized expert 0 output: first 5 values:\n");
    for (int j = 0; j < 5; j++) printf("  [%d] %.10f\n", j, out_q[j]);
    
    // Cos-sim between F32 and quantized
    double dot=0, n1=0, n2=0, max_diff=0;
    int max_idx = -1;
    for (int i = 0; i < D_MODEL; i++) {
        double d = (double)out_f32[i] - (double)out_q[i];
        if (fabs(d) > max_diff) { max_diff = fabs(d); max_idx = i; }
        dot += (double)out_f32[i] * (double)out_q[i];
        n1 += (double)out_f32[i] * (double)out_f32[i];
        n2 += (double)out_q[i] * (double)out_q[i];
    }
    printf("\nF32 vs Quantized expert 0:\n");
    printf("  cos-sim: %.10f\n", dot/(sqrt(n1)*sqrt(n2)));
    printf("  max-diff: %.10f at %d\n", max_diff, max_idx);
    if (max_idx >= 0) {
        printf("  F32[%d]=%.6f Q[%d]=%.6f\n", max_idx, out_f32[max_idx], max_idx, out_q[max_idx]);
    }
    
    free(gate_f32_expert0);
    free(up_f32_expert0);
    free(down_f32_expert0);
    free(emb);
    free(normed);
    free(norm_w);
    gguf_close(ctx);
    return 0;
}
