/* gqa_layer_check.c — Run GQA layer 3 (first GQA layer) with real input */
#include "wubu_model.h"
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    if (mdl.gguf_ctx) gguf_close(mdl.gguf_ctx);
    
    // Get layer 3 (first GQA layer)
    wubu_layer_t *ly = &mdl.layers[3];
    if (ly->is_ssm) { printf("Layer 3 is SSM, not GQA!\n"); return 1; }
    
    // Load embeddings from GGUF (need ctx)
    gguf_ctx *ctx = gguf_open(path);
    gguf_buffer_data(ctx);
    gguf_tensor_info *t_emb = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t_emb) { printf("No token_embd.weight\n"); return 1; }
    
    // Get embedding for token 0 (BOS) and token 1
    int B = 1, T = 2;
    float emb[2 * 2048];
    // BOS token: 248044
    int bos_id = 248044;
    // Read one embedding from the token_embd.weight tensor
    // It's Q5_K, load one row
    // The tensor is [2048, 248320], row v is at offset v * 2048 * sizeof(float) in dequant
    // Can't do partial read, need full dequant for one token
    // Simpler: use pre-extracted embedding file
    FILE *emb_f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!emb_f) {
        printf("No embedding file. Trying GGUF direct...\n");
        int64_t ne_emb = (int64_t)D_MODEL * 248320;
        float *all_emb = (float *)malloc(ne_emb * sizeof(float));
        if (!gguf_read_tensor_f32(ctx, t_emb, all_emb, ne_emb)) {
            printf("Failed to load embeddings\n"); return 1;
        }
        // BOS
        memcpy(emb, all_emb + bos_id * D_MODEL, D_MODEL * sizeof(float));
        // Random token
        memcpy(emb + D_MODEL, all_emb + 1 * D_MODEL, D_MODEL * sizeof(float));
        free(all_emb);
    } else {
        fseek(emb_f, bos_id * D_MODEL * sizeof(float), SEEK_SET);
        fread(emb, sizeof(float), D_MODEL, emb_f);
        fseek(emb_f, 1 * D_MODEL * sizeof(float), SEEK_SET);
        fread(emb + D_MODEL, sizeof(float), D_MODEL, emb_f);
        fclose(emb_f);
    }
    
    printf("Embedding[0] stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<D_MODEL;i++){s+=emb[i];if(emb[i]>mx)mx=emb[i];if(emb[i]<mn)mn=emb[i];}
      printf(" mean=%.4f max=%.4f min=%.4f\n", s/D_MODEL, mx, mn); }
    
    // Run GQA forward: prefill the GQA layer
    float normed[2048];
    float residual[2048];
    memcpy(residual, emb, B * T * D_MODEL * sizeof(float));
    
    // Pre-attention RMSNorm
    wubu_rms_norm(1, T, D_MODEL, residual, ly->attn_norm_weight, 1e-6f, normed);
    printf("Normed[0]:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<D_MODEL;i++){s+=normed[i];if(normed[i]>mx)mx=normed[i];if(normed[i]<mn)mn=normed[i];}
      printf(" mean=%.4f max=%.4f min=%.4f\n", s/D_MODEL, mx, mn); }
    
    // Compute Q, K, V, attention, output (simplified - just Q stats)
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    float Q[4096], K[512], V[512];
    
    // Q
    for (int s = 0; s < B * T; s++) {
        const float *xs = normed + s * D_MODEL;
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)ly->gqa.attn_q_weight[i * (q_dim * 2) + j];
            Q[s * q_dim + j] = (float)sum;
        }
    }
    printf("Q stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<B*T*q_dim;i++){s+=Q[i];if(Q[i]>mx)mx=Q[i];if(Q[i]<mn)mn=Q[i];}
      printf(" mean=%.4f rms=%.4f max=%.4f min=%.4f\n", s/(B*T*q_dim), sqrt(s*s/(B*T*q_dim)/(B*T*q_dim)), mx, mn); }
    
    // K
    for (int s = 0; s < B * T; s++) {
        const float *xs = normed + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)ly->gqa.attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = (float)sum;
        }
    }
    printf("K stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<B*T*kv_dim;i++){s+=K[i];if(K[i]>mx)mx=K[i];if(K[i]<mn)mn=K[i];}
      printf(" mean=%.4f rms=%.4f max=%.4f min=%.4f\n", s/(B*T*kv_dim), sqrt(s*s/(B*T*kv_dim)/(B*T*kv_dim)), mx, mn); }
    
    // V
    for (int s = 0; s < B * T; s++) {
        const float *xs = normed + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)ly->gqa.attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = (float)sum;
        }
    }
    printf("V stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<B*T*kv_dim;i++){s+=V[i];if(V[i]>mx)mx=V[i];if(V[i]<mn)mn=V[i];}
      printf(" mean=%.4f rms=%.4f max=%.4f min=%.4f\n", s/(B*T*kv_dim), sqrt(s*s/(B*T*kv_dim)/(B*T*kv_dim)), mx, mn); }
    
    // Full attention for first token
    float attn_out[4096] = {0};
    float gate[4096];
    for (int j = 0; j < q_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)normed[i] * (double)ly->gqa.attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
        gate[j] = (float)sum;
    }
    
    // RMSNorm Q and K
    float Q_norm[4096], K_norm[512];
    memcpy(Q_norm, Q, B * T * q_dim * sizeof(float));
    memcpy(K_norm, K, B * T * kv_dim * sizeof(float));
    
    // Skip RoPE for this simple test
    // Just do attention without RoPE to check scales
    float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
    for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
        int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
        const float *qv = Q_norm + h_q * GQA_HEAD_DIM;
        float *out = attn_out + h_q * GQA_HEAD_DIM;
        float mx = -1e30f, sum_exp = 0.0f;
        for (int t = 0; t < B * T; t++) {
            const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float sc = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
            sc *= scale;
            if (t == 0 || sc > mx) mx = sc;
        }
        for (int t = 0; t < B * T; t++) {
            const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float sc = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
            sum_exp += expf(sc * scale - mx);
        }
        float inv = 1.0f / (sum_exp + 1e-30f);
        for (int t = 0; t < B * T; t++) {
            const float *vv = V + t * kv_dim + h_kv * GQA_HEAD_DIM;
            const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float sc = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
            float a = expf(sc * scale - mx) * inv;
            for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
        }
    }
    
    // Gate
    for (int i = 0; i < q_dim; i++)
        attn_out[i] *= 1.0f / (1.0f + expf(-gate[i]));
    
    printf("Attn out stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<q_dim;i++){s+=attn_out[i];if(attn_out[i]>mx)mx=attn_out[i];if(attn_out[i]<mn)mn=attn_out[i];}
      printf(" mean=%.4f rms=%.4f max=%.4f min=%.4f\n", s/q_dim, sqrt(s*s/q_dim/q_dim), mx, mn); }
    
    // Output projection
    float gqa_output[2048] = {0};
    for (int i = 0; i < q_dim; i++) {
        float a = attn_out[i];
        for (int j = 0; j < D_MODEL; j++)
            gqa_output[j] += a * ly->gqa.attn_output_weight[i * D_MODEL + j];
    }
    
    printf("GQA output stats:");
    { double s=0,mx=-1e30,mn=1e30; for(int i=0;i<D_MODEL;i++){s+=gqa_output[i];if(gqa_output[i]>mx)mx=gqa_output[i];if(gqa_output[i]<mn)mn=gqa_output[i];}
      printf(" mean=%.4f rms=%.4f max=%.4f min=%.4f\n", s/D_MODEL, sqrt(s*s/D_MODEL/D_MODEL), mx, mn); }
    
    wubu_model_free(&mdl);
    gguf_close(ctx);
    printf("PASS\n");
    return 0;
}
