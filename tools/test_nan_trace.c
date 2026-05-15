/**
 * test_nan_trace.c — Find first NaN source in model forward
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    wubu_model_t model;
    if (!wubu_model_init(&model, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;

    // Load text embeddings
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { perror("fopen"); return 1; }
    float embd[8*2048];
    fread(embd, sizeof(float), 8*2048, f);
    fclose(f);

    int B = 1, T = 8, N = B*T, D = D_MODEL;
    float *x = (float *)malloc(N * D * sizeof(float));
    memcpy(x, embd, N * D * sizeof(float));

    printf("Tracing NaN through 40 layers...\n");
    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];

        // Pre-attention RMSNorm
        float *normed = (float *)malloc(N * D * sizeof(float));
        wubu_rms_norm(B, T, D, x, layer->attn_norm_weight, 1e-6f, normed);

        // Check normed for NaN
        for (int i = 0; i < N*D; i++) {
            if (isnan(normed[i])) {
                printf("  L%d RMSNorm NaN at [%d,%d]\n", l, i/D, i%D);
                goto next_layer;
            }
        }

        // Attention
        float *attn_out = (float *)malloc(N * D * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ssm_state, conv_state, attn_out);
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out);
        }

        // Check attn_out for NaN
        int first_nan = -1;
        for (int i = 0; i < N*D; i++) {
            if (isnan(attn_out[i])) { first_nan = i; break; }
        }
        if (first_nan >= 0) {
            printf("  L%d (%s) FIRST NaN at [%d,%d] = %f\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   first_nan / D, first_nan % D, attn_out[first_nan]);
            // Check normed right before
            int check = first_nan;
            printf("    normed[%d]=%f attn_out[%d-4..+4]:", check, normed[check]);
            for (int j = -4; j <= 4; j++) {
                int idx = first_nan + j;
                if (idx >= 0 && idx < N*D) printf(" %.4f", attn_out[idx]);
            }
            printf("\n");

            // For GQA, check Q/K/V projections
            if (!layer->is_ssm) {
                printf("    Checking GQA intermediate values...\n");
                int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM; // 4096
                int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM; // 512
                float *Q_raw = (float *)malloc(N * q_dim * sizeof(float));
                // Project Q
                for (int s = 0; s < N; s++) {
                    for (int j = 0; j < q_dim; j++) {
                        double sum = 0;
                        for (int k = 0; k < D; k++)
                            sum += (double)normed[s*D + k] * (double)layer->gqa.attn_q_weight[k*q_dim + j];
                        Q_raw[s*q_dim + j] = (float)sum;
                    }
                }
                // Check Q for NaN
                for (int i = 0; i < N*q_dim; i++) {
                    if (isnan(Q_raw[i])) {
                        printf("    Q_raw NaN at [%d,%d]\n", i/q_dim, i%q_dim);
                        break;
                    }
                }
                // Check if NaN at specific positions
                int tgt_tok = first_nan / D;
                int tgt_dim = first_nan % D;
                printf("    Target at [t=%d,d=%d]: normed=%.4f\n", tgt_tok, tgt_dim, normed[tgt_tok*D + tgt_dim]);
                free(Q_raw);
            }
            goto next_layer;
        }

        // Residual
        for (int i = 0; i < N * D; i++) x[i] += attn_out[i];

        // Post-attention RMSNorm
        float *normed2 = (float *)malloc(N * D * sizeof(float));
        wubu_rms_norm(B, T, D, x, layer->post_attn_norm_weight, 1e-6f, normed2);

        // MoE (pass-through)
        for (int i = 0; i < N * D; i++) x[i] += normed2[i];

        free(normed);
        free(attn_out);
        free(normed2);
    }

    printf("All 40 layers clean (no NaN)\n");
    wubu_model_free(&model);
    free(x);
    return 0;

next_layer:
    wubu_model_free(&model);
    free(x);
    return 1;
}
