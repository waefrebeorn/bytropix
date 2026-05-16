#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Test SSM Layer 0 in isolation.
 * Loads layer 0 SSM weights, computes embedding for token 9419,
 * runs wubu_ssm_forward, dumps output to /tmp/our_ssm_l0.bin
 */
int main(int argc, char **argv) {
    const char *model_path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Open GGUF
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { fprintf(stderr, "Failed to open model\n"); return 1; }
    
    // Load embedding table (token_embd.weight)
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "No token_embd.weight\n"); return 1; }
    int64_t ne = t->dims[0] * t->dims[1];
    float *embd = (float *)malloc(ne * sizeof(float));
    if (!embd || gguf_read_tensor_f32(ctx, t, embd, ne) <= 0) {
        fprintf(stderr, "Failed to read embeddings\n"); return 1;
    }
    
    // Load SSM Layer 0 weights
    ssm_layer_weights w;
    memset(&w, 0, sizeof(w));
    char name[256];
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;  // 2048*2 + 4096 = 8192
    
    // attn_qkv.weight
    snprintf(name, sizeof(name), "blk.0.attn_qkv.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.attn_qkv_weight, D_MODEL * qkv_dim);
    printf("attn_qkv: [%ld, %ld] -> f32\n", (long)t->dims[0], (long)t->dims[1]);
    
    // attn_gate.weight
    snprintf(name, sizeof(name), "blk.0.attn_gate.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.attn_gate_weight, D_MODEL * VALUE_DIM);
    printf("attn_gate: [%ld, %ld]\n", (long)t->dims[0], (long)t->dims[1]);
    
    // ssm_beta.weight
    snprintf(name, sizeof(name), "blk.0.ssm_beta.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_beta_weight, D_MODEL * DT_RANK);
    printf("ssm_beta: [%ld, %ld]\n", (long)t->dims[0], (long)t->dims[1]);
    
    // ssm_alpha.weight
    snprintf(name, sizeof(name), "blk.0.ssm_alpha.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_alpha_weight, D_MODEL * DT_RANK);
    printf("ssm_alpha: [%ld, %ld]\n", (long)t->dims[0], (long)t->dims[1]);
    
    // ssm_dt.bias
    snprintf(name, sizeof(name), "blk.0.ssm_dt.bias");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_dt_bias, DT_RANK);
    printf("ssm_dt.bias: %d elems\n", DT_RANK);
    
    // ssm_a
    snprintf(name, sizeof(name), "blk.0.ssm_a");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_a, DT_RANK);
    printf("ssm_a: [%ld] elems=%lld\n", (long)t->dims[0], (long long)(t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1)));
    
    // ssm_conv1d.weight
    snprintf(name, sizeof(name), "blk.0.ssm_conv1d.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);
    printf("ssm_conv1d: [%ld, %ld]\n", (long)t->dims[0], (long)t->dims[1]);
    
    // ssm_norm.weight
    snprintf(name, sizeof(name), "blk.0.ssm_norm.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_norm_weight, SSM_D_STATE);
    printf("ssm_norm: %ld elems\n", (long)t->dims[0]);
    
    // ssm_out.weight
    int v_dim = VALUE_DIM;
    snprintf(name, sizeof(name), "blk.0.ssm_out.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) return 1;
    w.ssm_out_weight = (float *)malloc(v_dim * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w.ssm_out_weight, v_dim * D_MODEL);
    printf("ssm_out: [%ld, %ld]\n", (long)t->dims[0], (long)t->dims[1]);
    
    // Get embedding for token 9419 ("Hello")
    int token = 9419;
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(x, embd + token * D_MODEL, D_MODEL * sizeof(float));
    
    // Verify embedding
    float mn=1e30,mx=-1e30,sum=0;
    for (int i = 0; i < D_MODEL; i++) {
        if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; sum += fabsf(x[i]);
    }
    printf("Emb token %d: mean|%.4f max=%.4f min=%.4f\n", token, sum/D_MODEL, mx, mn);
    
    // Run SSM forward (1 batch, 1 token)
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *output = (float *)malloc(D_MODEL * sizeof(float));
    
    wubu_ssm_forward(x, 1, 1, &w, ssm_state, conv_state, output);
    
    // Dump output to file
    FILE *f = fopen("/tmp/our_ssm_l0.bin", "wb");
    if (f) { fwrite(output, sizeof(float), D_MODEL, f); fclose(f); }
    
    // Stats
    float omx=-1e30, omn=1e30, osm=0, osm2=0;
    for (int i = 0; i < D_MODEL; i++) {
        if (output[i] > omx) omx = output[i];
        if (output[i] < omn) omn = output[i];
        osm += output[i];
        osm2 += output[i] * output[i];
    }
    printf("SSM L0 output: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
           omn, omx, osm/D_MODEL, sqrtf(osm2/D_MODEL));
    printf("  first 5: %.4f %.4f %.4f %.4f %.4f\n",
           output[0], output[1], output[2], output[3], output[4]);
    
    // Cleanup
    free(embd);
    free(w.attn_qkv_weight); free(w.attn_gate_weight);
    free(w.ssm_beta_weight); free(w.ssm_alpha_weight);
    free(w.ssm_dt_bias); free(w.ssm_a);
    free(w.ssm_conv1d_weight); free(w.ssm_norm_weight);
    free(w.ssm_out_weight);
    free(x); free(ssm_state); free(conv_state); free(output);
    gguf_close(ctx);
    return 0;
}
