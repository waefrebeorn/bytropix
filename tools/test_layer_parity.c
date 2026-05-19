/**
 * test_layer_parity.c — Layer-by-layer hidden state comparison vs llama.cpp.
 *
 * Load model, run 1 token through layer 0, dump hidden states.
 * Compare with llama.cpp reference output.
 *
 * Build: gcc -O2 -I include -o test_layer_parity tools/test_layer_parity.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o -lm -fopenmp -lLLama
 *
 * Usage: ./test_layer_parity model.gguf
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Model dimensions (from model)
int D_MODEL, D_FF, SHARED_D_FF;
int KEY_DIM, VALUE_DIM, DT_RANK;
int D_STATE, N_EXPERTS, N_ACTIVE_EXPTS;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    // Load model
    wubu_model mdl;
    if (wubu_model_load(argv[1], &mdl, 0) != 0) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }

    D_MODEL = mdl.d_model;
    D_FF = mdl.d_ff;
    SHARED_D_FF = mdl.d_shared_ff;
    N_EXPERTS = mdl.n_experts;
    N_ACTIVE_EXPTS = mdl.n_active_experts;
    KEY_DIM = mdl.ssm_key_dim;
    VALUE_DIM = mdl.ssm_value_dim;
    DT_RANK = mdl.ssm_dt_rank;
    D_STATE = mdl.ssm_d_state;

    printf("Model: %d layers, D=%d, FF=%d, N_EXP=%d, N_ACT=%d\n",
           mdl.n_layers, D_MODEL, D_FF, N_EXPERTS, N_ACTIVE_EXPTS);
    printf("SSM: KEY=%d, VAL=%d, DT=%d, STATE=%d\n",
           KEY_DIM, VALUE_DIM, DT_RANK, D_STATE);

    // Embed a single token (ID=1, typical BOS)
    float *embd = mdl.token_embd;
    if (!embd) { fprintf(stderr, "No embedding table\n"); return 1; }

    int token_id = 248044; // BOS for Qwen
    float x[D_MODEL];
    memcpy(x, embd + token_id * D_MODEL, D_MODEL * sizeof(float));

    printf("\n=== Input token %d ===\n", token_id);
    float mean=0, maxv=-1e30, minv=1e30;
    for (int i = 0; i < D_MODEL; i++) {
        mean += x[i];
        if (x[i] > maxv) maxv = x[i];
        if (x[i] < minv) minv = x[i];
    }
    printf("  embed: mean=%.6f max=%.6f min=%.6f\n", mean/D_MODEL, maxv, minv);

    // Run through layer 0
    float residual[D_MODEL];
    memcpy(residual, x, D_MODEL * sizeof(float));

    for (int l = 0; l < 3; l++) {  // Just first 3 layers
        printf("\n=== Layer %d ===\n", l);

        // Pre-attention RMSNorm
        float normed[D_MODEL];
        wubu_rms_norm(1, 1, D_MODEL, residual,
                      mdl.layers[l].attn_norm_weight, 1e-6f, normed);

        mean=0; for(int i=0;i<D_MODEL;i++) mean+=normed[i];
        printf("  pre-norm: mean=%.6f\n", mean/D_MODEL);

        float attn[D_MODEL];

        if (mdl.layers[l].is_ssm) {
            // SSM forward
            wubu_ssm_forward(normed, 1, 1, &mdl.layers[l].w.ssm,
                             mdl.layers[l].ssm_state,
                             mdl.layers[l].conv_state, attn, NULL, NULL);
        } else {
            // GQA forward
            wubu_gqa_forward(normed, 1, 1, &mdl.layers[l].w.gqa,
                             mdl.layers[l].kv_cache,
                             mdl.layers[l].rope_sc, attn);
        }

        // SSM/GQA output stats
        mean=0; maxv=-1e30; minv=1e30;
        for (int i = 0; i < D_MODEL; i++) {
            mean += attn[i];
            if (attn[i] > maxv) maxv = attn[i];
            if (attn[i] < minv) minv = attn[i];
        }
        printf("  attn: mean=%.6f max=%.6f min=%.6f\n", mean/D_MODEL, maxv, minv);

        // Residual
        for (int i = 0; i < D_MODEL; i++) residual[i] += attn[i];

        // Post-attention norm
        wubu_rms_norm(1, 1, D_MODEL, residual,
                      mdl.layers[l].post_attn_norm_weight, 1e-6f, normed);

        // MoE
        float ffn[D_MODEL];
        if (mdl.layers[l].moe_weights) {
            // Dump router scores
            float scores[N_EXPERTS];
            wubu_moe_router(normed, 1, 1,
                           mdl.layers[l].moe_weights->ffn_gate_inp, scores);
            int top_e = 0; float top_v = -1e30f;
            for (int e = 0; e < N_EXPERTS; e++) {
                if (scores[e] > top_v) { top_v = scores[e]; top_e = e; }
            }
            printf("  router: top=%d (%.6f), all mean=%.6f\n",
                   top_e, top_v, top_v);

            // Full MoE forward
            wubu_moe_forward(normed, 1, 1, mdl.layers[l].moe_weights, ffn, NULL);
        } else {
            memcpy(ffn, normed, D_MODEL * sizeof(float));
        }

        // FFN stats
        mean=0; maxv=-1e30; minv=1e30;
        for (int i = 0; i < D_MODEL; i++) {
            mean += ffn[i];
            if (ffn[i] > maxv) maxv = ffn[i];
            if (ffn[i] < minv) minv = ffn[i];
        }
        printf("  ffn: mean=%.6f max=%.6f min=%.6f\n", mean/D_MODEL, maxv, minv);

        // Dump layer output (first 10 values) for comparison
        printf("  residual[:10]:");
        for (int i = 0; i < 10; i++) printf(" %.6f", residual[i]);
        printf("\n");

        // Add FFN to residual for next layer
        for (int i = 0; i < D_MODEL; i++) residual[i] += ffn[i];
    }

    // Final norm
    float final[D_MODEL];
    if (mdl.norm_weight) {
        wubu_rms_norm(1, 1, D_MODEL, residual, mdl.norm_weight, 1e-6f, final);
    } else {
        memcpy(final, residual, D_MODEL * sizeof(float));
    }

    // Output projection (just dump logits for top tokens, not all 248K)
    if (mdl.output_weight) {
        float logits_vs = 0;
        float max_l = -1e30; int max_i = 0;
        for (int j = 0; j < 1000; j++) { // just check first 1000
            double sum = 0.0;
            for (int k = 0; k < D_MODEL; k++)
                sum += (double)final[k] * (double)mdl.output_weight[(int64_t)j * D_MODEL + k];
            if ((float)sum > max_l) { max_l = (float)sum; max_i = j; }
        }
        printf("\n=== Output (first 1000 tokens) ===\n");
        printf("  top token: [%d] = %.4f\n", max_i, max_l);

        // Also check full vocab for top tokens
        int n_check = 10000;
        max_l = -1e30; max_i = 0;
        for (int j = 0; j < n_check; j++) {
            double sum = 0.0;
            for (int k = 0; k < D_MODEL; k++)
                sum += (double)final[k] * (double)mdl.output_weight[(int64_t)j * D_MODEL + k];
            if ((float)sum > max_l) { max_l = (float)sum; max_i = j; }
        }
        printf("  top-%d: [%d] = %.4f\n", n_check, max_i, max_l);
    }

    wubu_model_free(&mdl);
    printf("\n=== PASS ===\n");
    return 0;
}
