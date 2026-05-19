#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Load embedding
    float *emb = (float *)malloc(2048 * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    fseek(f, 248044LL * 2048 * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), 2048, f);
    fclose(f);
    double sum_sq = 0;
    for (int i = 0; i < 2048; i++) sum_sq += (double)emb[i] * emb[i];
    double r = 1.0 / sqrt(sum_sq / 2048 + 1e-6);
    for (int i = 0; i < 2048; i++) emb[i] = (float)(emb[i] * r);
    
    // Load F32 gate_shexp
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_shexp.weight");
    int64_t gate_elems = (int64_t)2048 * 512;
    float *gate_f32 = (float *)malloc(gate_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, gate_f32, gate_elems);
    
    // F32 gate matmul
    float gate_f32_out[512];
    for (int j = 0; j < 512; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++) sum += (double)emb[k] * (double)gate_f32[k + j * 2048];
        gate_f32_out[j] = (float)sum;
    }
    
    // Quantized gate matmul
    const uint8_t *gate_q = blob + t->data_offset;
    int gate_type = t->ggml_type;
    float gate_q_out[512];
    quantized_matmul(emb, gate_q, gate_type, 2048, 512, 0, gate_q_out);
    
    // Compare gate outputs
    double dot=0, n1=0, n2=0, md=0; int mi=-1;
    for (int i = 0; i < 512; i++) {
        double d = gate_f32_out[i] - gate_q_out[i];
        if (fabs(d) > md) { md = fabs(d); mi = i; }
        dot += (double)gate_f32_out[i] * (double)gate_q_out[i];
        n1 += (double)gate_f32_out[i] * (double)gate_f32_out[i];
        n2 += (double)gate_q_out[i] * (double)gate_q_out[i];
    }
    printf("GATE Q5_K: cos=%.10f md=%.6f@%d f32=%.4f q=%.4f\n", 
           dot/(sqrt(n1)*sqrt(n2)), md, mi, gate_f32_out[mi], gate_q_out[mi]);
    printf("First 10 F32 gate: "); for(int i=0;i<5;i++) printf("%.4f ", gate_f32_out[i]); printf("\n");
    printf("First 10 Q gate:  "); for(int i=0;i<5;i++) printf("%.4f ", gate_q_out[i]); printf("\n");
    
    // Load F32 down_shexp and test
    t = gguf_find_tensor(ctx, "blk.0.ffn_down_shexp.weight");
    int64_t down_elems = (int64_t)512 * 2048;
    float *down_f32 = (float *)malloc(down_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, down_f32, down_elems);
    
    // Use gate_f32_out as activation for down test
    float *act = gate_f32_out; // use F32 gate result as activation
    float down_f32_out[2048];
    for (int j = 0; j < 2048; j++) {
        double sum = 0;
        for (int k = 0; k < 512; k++) sum += (double)act[k] * (double)down_f32[k + j * 512];
        down_f32_out[j] = (float)sum;
    }
    
    const uint8_t *down_q = blob + t->data_offset;
    int down_type = t->ggml_type;
    float down_q_out[2048];
    quantized_matmul(act, down_q, down_type, 512, 2048, 0, down_q_out);
    
    dot=0; n1=0; n2=0; md=0; mi=-1;
    for (int i = 0; i < 2048; i++) {
        double d = down_f32_out[i] - down_q_out[i];
        if (fabs(d) > md) { md = fabs(d); mi = i; }
        dot += (double)down_f32_out[i] * (double)down_q_out[i];
        n1 += (double)down_f32_out[i] * (double)down_f32_out[i];
        n2 += (double)down_q_out[i] * (double)down_q_out[i];
    }
    printf("\nDOWN Q6_K: cos=%.10f md=%.6f@%d f32=%.4f q=%.4f\n",
           dot/(sqrt(n1)*sqrt(n2)), md, mi, down_f32_out[mi], down_q_out[mi]);
    printf("First 10 F32 down: "); for(int i=0;i<5;i++) printf("%.4f ", down_f32_out[i]); printf("\n");
    printf("First 10 Q down:  "); for(int i=0;i<5;i++) printf("%.4f ", down_q_out[i]); printf("\n");
    
    free(gate_f32); free(down_f32); free(emb);
    gguf_close(ctx);
    return 0;
}
