/**
 * dump_rmsnorm.c — Verify RMSNorm: dump normed input, weights, and output.
 * Build: gcc -O2 -I include -o dump_rmsnorm tools/dump_rmsnorm.c \
 *     src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *     src/wubu_moe.o src/wubu_tokenizer.o src/qlearner.o -lm -fopenmp
 * Usage: ./dump_rmsnorm model.gguf
 */
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed open\n"); return 1; }
    
    // BOS embedding
    int D = D_MODEL; // 2048
    float *embd = (float *)malloc(D * sizeof(float));
    
    // Read token_embd.weight
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    float *all_embd = (float *)malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all_embd, n_elems);
    
    // BOS token
    int bos_id = 248044;
    memcpy(embd, all_embd + bos_id * D, D * sizeof(float));
    printf("BOS embd rms=%.10f\n", sqrtf(1.0f/D * ({double s=0;for(int i=0;i<D;i++)s+=embd[i]*embd[i];s;})));
    
    // Dump embd
    FILE *f = fopen("/tmp/test_embd.bin", "wb");
    fwrite(embd, sizeof(float), D, f); fclose(f);
    
    // Read attn_norm.weight for layer 0
    float norm_w[D];
    t = gguf_find_tensor(ctx, "blk.0.attn_norm.weight");
    gguf_read_tensor_f32(ctx, t, norm_w, D);
    
    // Dump norm weight
    f = fopen("/tmp/test_norm_w.bin", "wb");
    fwrite(norm_w, sizeof(float), D, f); fclose(f);
    
    // Our RMSNorm
    float normed[D];
    wubu_rms_norm(1, 1, D, embd, norm_w, 1e-6f, normed);
    printf("Our rms_norm rms=%.10f\n", sqrtf(1.0f/D * ({double s=0;for(int i=0;i<D;i++)s+=normed[i]*normed[i];s;})));
    
    // Dump normed
    f = fopen("/tmp/test_normed.bin", "wb");
    fwrite(normed, sizeof(float), D, f); fclose(f);
    
    // ---- SSM forward ----
    ssm_layer_weights sw;
    memset(&sw, 0, sizeof(sw));
    char name[256];
    
    // Load SSM weights for layer 0
    snprintf(name, sizeof(name), "blk.0.attn_qkv.weight");
    gguf_tensor_info *t_qkv = gguf_find_tensor(ctx, name);
    if (!t_qkv) {
        // Maybe the tensor has a different name
        fprintf(stderr, "Trying alternate tensor name...\n");
        // Just continue without SSM
    } else {
        printf("'%s' dims: %d [", name, t_qkv->n_dims);
        for (int d = 0; d < t_qkv->n_dims; d++) printf("%lld%s", (long long)t_qkv->dims[d], d+1<t_qkv->n_dims?",":"");
        printf("] type=%d\n", t_qkv->n_dims);
        fflush(stdout);
        
        int64_t n_qkv = 1;
        for (int d = 0; d < t_qkv->n_dims; d++) n_qkv *= t_qkv->dims[d];
        printf("  n_elems=%lld\n", (long long)n_qkv);
        fflush(stdout);
        
        sw.attn_qkv_weight = (float *)malloc(n_qkv * sizeof(float));
        if (!gguf_read_tensor_f32(ctx, t_qkv, sw.attn_qkv_weight, n_qkv)) {
            fprintf(stderr, "Failed to read %s\n", name);
        }
    }
    
    snprintf(name, sizeof(name), "blk.0.attn_gate.weight");
    t = gguf_find_tensor(ctx, name);
    sw.attn_gate_weight = (float *)malloc(D * VALUE_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.attn_gate_weight, D * VALUE_DIM);
    
    snprintf(name, sizeof(name), "blk.0.ssm_conv1d.weight");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_conv1d_weight = (float *)malloc(CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_conv1d_weight, CONV_DIM);
    
    snprintf(name, sizeof(name), "blk.0.ssm_dt.bias");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_dt_bias = (float *)malloc(SSM_V_HEADS * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_dt_bias, SSM_V_HEADS);
    
    snprintf(name, sizeof(name), "blk.0.ssm_a");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_a = (float *)malloc(SSM_V_HEADS * SSM_D_STATE * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_a, SSM_V_HEADS * SSM_D_STATE);
    
    snprintf(name, sizeof(name), "blk.0.ssm_beta.weight");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_beta_weight = (float *)malloc(SSM_V_HEADS * CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_beta_weight, SSM_V_HEADS * CONV_DIM);
    
    snprintf(name, sizeof(name), "blk.0.ssm_alpha.weight");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_alpha_weight = (float *)malloc(SSM_V_HEADS * CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_alpha_weight, SSM_V_HEADS * CONV_DIM);
    
    snprintf(name, sizeof(name), "blk.0.ssm_norm.weight");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_norm_weight = (float *)malloc(CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_norm_weight, CONV_DIM);
    
    snprintf(name, sizeof(name), "blk.0.ssm_out.weight");
    t = gguf_find_tensor(ctx, name);
    sw.ssm_out_weight = (float *)malloc(D * CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, sw.ssm_out_weight, D * CONV_DIM);
    
    // Run SSM forward
    float attn[D];
    memset(attn, 0, D * sizeof(float));
    
    // SSM states
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    
    wubu_ssm_forward(normed, 1, 1, &sw, ssm_state, conv_state, attn);
    printf("SSM attn rms=%.10f\n", sqrtf(1.0f/D * ({double s=0;for(int i=0;i<D;i++)s+=attn[i]*attn[i];s;})));
    
    // Dump SSM output
    f = fopen("/tmp/test_ssm_out.bin", "wb");
    fwrite(attn, sizeof(float), D, f); fclose(f);
    
    // Compare: residual = emb + attn
    float residual[D];
    for (int i = 0; i < D; i++) residual[i] = embd[i] + attn[i];
    printf("Residual after layer 0 rms=%.10f\n", sqrtf(1.0f/D * ({double s=0;for(int i=0;i<D;i++)s+=residual[i]*residual[i];s;})));
    
    free(embd); free(all_embd);
    free(sw.attn_qkv_weight); free(sw.attn_gate_weight);
    free(sw.ssm_conv1d_weight); free(sw.ssm_dt_bias);
    free(sw.ssm_a); free(sw.ssm_beta_weight); free(sw.ssm_alpha_weight);
    free(sw.ssm_norm_weight); free(sw.ssm_out_weight);
    free(ssm_state); free(conv_state);
    gguf_close(ctx);
    printf("Done. /tmp/test_*.bin\n");
    return 0;
}
