/**
 * test_nan_xrange.c — Check x range, skip MoE properly
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    printf("Starting test_nan_xrange...\n"); fflush(stdout);
    wubu_model_t model;
    if (!wubu_model_init(&model, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    printf("Model loaded.\n"); fflush(stdout);

    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { perror("fopen"); return 1; }
    float embd[8*2048];
    fread(embd, sizeof(float), 8*2048, f);
    fclose(f);
    printf("Embeddings loaded.\n"); fflush(stdout);

    int B = 1, T = 8, N = B*T, D = D_MODEL;
    float *x = (float *)malloc(N * D * sizeof(float));
    if (!x) { printf("malloc x failed\n"); return 1; }
    memcpy(x, embd, N * D * sizeof(float));
    printf("Starting 40-layer forward...\n"); fflush(stdout);

    for (int l = 0; l < model.n_layers; l++) {
        if (l % 10 == 0) { printf("  Layer %d...\n", l); fflush(stdout); }
        wubu_layer_t *layer = &model.layers[l];
        
        float *normed = (float *)malloc(N * D * sizeof(float));
        if (!normed) { printf("malloc normed failed at L%d\n", l); return 1; }
        wubu_rms_norm(B, T, D, x, layer->attn_norm_weight, 1e-6f, normed);

        float *attn_out = (float *)malloc(N * D * sizeof(float));
        if (!attn_out) { printf("malloc attn_out failed at L%d\n", l); return 1; }
        if (layer->is_ssm) {
            float *ss = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *cs = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ss, cs, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }
        for (int i = 0; i < N*D; i++) x[i] += attn_out[i];

        float *normed2 = (float *)malloc(N * D * sizeof(float));
        if (!normed2) { printf("malloc normed2 failed at L%d\n", l); return 1; }
        wubu_rms_norm(B, T, D, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        // No MoE — just pass through (same as original pass-through path)
        for (int i = 0; i < N*D; i++) x[i] += normed2[i];

        free(normed); free(attn_out); free(normed2);
        if (l % 10 == 0) printf("    Done.\n");
    }

    // Final RMSNorm
    if (model.norm_weight) {
        wubu_rms_norm(B, T, D, x, model.norm_weight, 1e-6f, x);
    }

    float mn=1e30, mx=-1e30; int nan_c = 0, inf_c = 0;
    for (int i = 0; i < N*D; i++) {
        if (x[i] < mn) mn=x[i]; if (x[i] > mx) mx=x[i];
        if (isnan(x[i])) nan_c++; if (isinf(x[i])) inf_c++;
    }
    printf("\nFinal x: [%.4f, %.4f] NaN=%d Inf=%d\n", mn, mx, nan_c, inf_c);
    printf("x[0:8]:"); for (int i=0;i<8;i++) printf(" %.4f", x[i]); printf("\n");

    // Check output_weight
    if (model.output_weight) {
        mn=1e30; mx=-1e30; nan_c = 0;
        int ws = D * model.vocab_size;
        for (int i = 0; i < ws; i++) { if (model.output_weight[i] < mn) mn=model.output_weight[i]; if (model.output_weight[i] > mx) mx=model.output_weight[i]; if (isnan(model.output_weight[i])) nan_c++; }
        printf("output_weight [%dx%d]: [%.4f, %.4f] NaN=%d\n", D, model.vocab_size, mn, mx, nan_c);
    }

    // Compute logit for first token and find NaN
    printf("output_weight pointer: %p, vocab_size: %d\n", (void*)model.output_weight, model.vocab_size); fflush(stdout);
    const float *h = x;
    printf("Checking x and output_weight validity...\n"); fflush(stdout);
    if (!model.output_weight) { printf("output_weight is NULL!\n"); return 1; }
    if (!h) { printf("h is NULL!\n"); return 1; }
    
    printf("output_weight[0] = %f\n", model.output_weight[0]); fflush(stdout);
    printf("output_weight[0*248320+0] = %f\n", model.output_weight[0 * model.vocab_size + 0]); fflush(stdout);
    printf("Starting output projection...\n"); fflush(stdout);
    
    int nan_vocab = -1;
    const int vocab_size = model.vocab_size;
    for (int j = 0; j < vocab_size && nan_vocab < 0; j++) {
        if (j % 50000 == 0) { printf("  vocab %d...\n", j); fflush(stdout); }
        double sum = 0;
        int overflow = 0;
        for (int k = 0; k < D && !overflow; k++) {
            double term = (double)h[k] * (double)model.output_weight[k * model.vocab_size + j];
            sum += term;
            if (!isfinite((float)sum)) overflow = 1;
        }
        if (overflow) { nan_vocab = j; }
    }
    printf("First NaN output at vocab=%d\n", nan_vocab);
    if (nan_vocab >= 0) {
        double sum = 0;
        for (int k = 0; k < D; k++) {
            double term = (double)h[k] * (double)model.output_weight[k * model.vocab_size + nan_vocab];
            sum += term;
            if (!isfinite((float)sum)) {
                printf("  Overflow at k=%d, h=%.4f w=%+.6f term=%.4e sum=%.4e\n",
                       k, h[k], model.output_weight[k * model.vocab_size + nan_vocab], term, sum);
                break;
            }
        }
    }

    wubu_model_free(&model);
    free(x);
    return 0;
}
