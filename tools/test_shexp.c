#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("fail\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    // Get shared expert tensor info
    gguf_tensor_info *t;
    char name[256];
    
    snprintf(name, sizeof(name), "blk.0.ffn_gate_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_gate_shexp: type=%d offset=%llu dims=%llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
    int gate_type = t->ggml_type;
    const uint8_t *gate_q = blob + t->data_offset;
    
    snprintf(name, sizeof(name), "blk.0.ffn_up_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_up_shexp: type=%d offset=%llu dims=%llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
    const uint8_t *up_q = blob + t->data_offset;
    
    snprintf(name, sizeof(name), "blk.0.ffn_down_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    printf("ffn_down_shexp: type=%d offset=%llu dims=%llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
    int down_type = t->ggml_type;
    const uint8_t *down_q = blob + t->data_offset;
    
    // Load embedding
    float *emb = (float *)malloc(2048 * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    fseek(f, 248044LL * 2048 * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), 2048, f);
    fclose(f);
    
    // RMS norm
    double sum_sq = 0;
    for (int i = 0; i < 2048; i++) sum_sq += (double)emb[i] * emb[i];
    double r = 1.0 / sqrt(sum_sq / 2048 + 1e-6);
    for (int i = 0; i < 2048; i++) emb[i] = (float)(emb[i] * r);
    
    // Load F32 shared expert for comparison
    int64_t gate_elems = (int64_t)2048 * 512;
    float *gate_f32 = (float *)malloc(gate_elems * sizeof(float));
    snprintf(name, sizeof(name), "blk.0.ffn_gate_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    gguf_read_tensor_f32(ctx, t, gate_f32, gate_elems);
    
    float *up_f32 = (float *)malloc(gate_elems * sizeof(float));
    snprintf(name, sizeof(name), "blk.0.ffn_up_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    gguf_read_tensor_f32(ctx, t, up_f32, gate_elems);
    
    int64_t down_elems = (int64_t)512 * 2048;
    float *down_f32 = (float *)malloc(down_elems * sizeof(float));
    snprintf(name, sizeof(name), "blk.0.ffn_down_shexp.weight");
    t = gguf_find_tensor(ctx, name);
    gguf_read_tensor_f32(ctx, t, down_f32, down_elems);
    
    // F32 shared expert forward
    float gate_f32_out[512], up_f32_out[512], act_f32[512], out_f32[2048];
    for (int j = 0; j < 512; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++) sum += (double)emb[k] * (double)gate_f32[k + j * 2048];
        gate_f32_out[j] = (float)sum;
    }
    for (int j = 0; j < 512; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++) sum += (double)emb[k] * (double)up_f32[k + j * 2048];
        up_f32_out[j] = (float)sum;
    }
    for (int j = 0; j < 512; j++) {
        float g = gate_f32_out[j];
        act_f32[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * up_f32_out[j];
    }
    for (int j = 0; j < 2048; j++) {
        double sum = 0;
        for (int k = 0; k < 512; k++) sum += (double)act_f32[k] * (double)down_f32[k + j * 512];
        out_f32[j] = (float)sum;
    }
    
    printf("\nF32 shared expert output (first 10):\n");
    for (int j = 0; j < 10; j++) printf("  [%d] %.10f\n", j, out_f32[j]);
    
    // Quantized shared expert forward
    float q_gate[512], q_up[512], q_act[512], q_out[2048];
    quantized_matmul(emb, gate_q, gate_type, 2048, 512, 0, q_gate);
    quantized_matmul(emb, up_q, gate_type, 2048, 512, 0, q_up);
    for (int j = 0; j < 512; j++) {
        float g = q_gate[j];
        q_act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * q_up[j];
    }
    quantized_matmul(q_act, down_q, down_type, 512, 2048, 0, q_out);
    
    printf("\nQuantized shared expert output (first 10):\n");
    for (int j = 0; j < 10; j++) printf("  [%d] %.10f\n", j, q_out[j]);
    
    // Cos-sim
    double dot=0, n1=0, n2=0, md=0; int mi=-1;
    for (int i = 0; i < 2048; i++) {
        double d = out_f32[i] - q_out[i];
        if (fabs(d) > md) { md = fabs(d); mi = i; }
        dot += (double)out_f32[i] * (double)q_out[i];
        n1 += (double)out_f32[i] * (double)out_f32[i];
        n2 += (double)q_out[i] * (double)q_out[i];
    }
    printf("\nF32 vs Quantized shared expert: cos=%.10f max-diff=%.10f\n", dot/(sqrt(n1)*sqrt(n2)), md);
    
    free(gate_f32); free(up_f32); free(down_f32);
    free(emb);
    gguf_close(ctx);
    return 0;
}
