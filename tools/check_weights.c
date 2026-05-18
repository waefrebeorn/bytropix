// Check SSM output weight and other weights for magnitude
#include "wubu_model.h"
#include <stdio.h>
#include <math.h>

int main() {
    wubu_model_t model;
    if (!wubu_model_init(&model, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) {
        fprintf(stderr, "FAIL load model\n");
        return 1;
    }
    for (int l = 0; l < model.n_layers && l < 2; l++) {
        if (model.layers[l].is_ssm) {
            printf("\n=== Layer %d SSM ===\n", l);

            // ssm_out weight [VALUE_DIM, D_MODEL]
            int n_out = VALUE_DIM * D_MODEL;
            float omax = -1e30f, omin = 1e30f;
            for (int i = 0; i < n_out; i++) {
                float v = model.layers[l].ssm.ssm_out_weight[i];
                if (v > omax) omax = v;
                if (v < omin) omin = v;
            }
            printf("  ssm_out [%dx%d]: min=%.2f max=%.2f\n", VALUE_DIM, D_MODEL, omin, omax);

            // ssm_norm [SSM_D_STATE]
            float nmax = -1e30f, nmin = 1e30f;
            for (int i = 0; i < SSM_D_STATE; i++) {
                float v = model.layers[l].ssm.ssm_norm_weight[i];
                if (v > nmax) nmax = v;
                if (v < nmin) nmin = v;
            }
            printf("  ssm_norm [%d]: min=%.2f max=%.2f\n", SSM_D_STATE, nmin, nmax);

            // ssm_conv1d [CONV_KERNEL, CONV_DIM]
            float cmax = -1e30f, cmin = 1e30f;
            for (int i = 0; i < CONV_KERNEL * CONV_DIM; i++) {
                float v = model.layers[l].ssm.ssm_conv1d_weight[i];
                if (v > cmax) cmax = v;
                if (v < cmin) cmin = v;
            }
            printf("  ssm_conv1d [%dx%d]: min=%.2f max=%.2f\n", CONV_KERNEL, CONV_DIM, cmin, cmax);

            // attn_qkv [D_MODEL, qkv_dim]
            int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
            float qmax = -1e30f, qmin = 1e30f;
            for (int i = 0; i < D_MODEL * qkv_dim; i++) {
                float v = model.layers[l].ssm.attn_qkv_weight[i];
                if (v > qmax) qmax = v;
                if (v < qmin) qmin = v;
            }
            printf("  attn_qkv [%dx%d]: min=%.2f max=%.2f\n", D_MODEL, qkv_dim, qmin, qmax);

            // attn_gate [D_MODEL, VALUE_DIM]
            float gmax = -1e30f, gmin = 1e30f;
            for (int i = 0; i < D_MODEL * VALUE_DIM; i++) {
                float v = model.layers[l].ssm.attn_gate_weight[i];
                if (v > gmax) gmax = v;
                if (v < gmin) gmin = v;
            }
            printf("  attn_gate [%dx%d]: min=%.2f max=%.2f\n", D_MODEL, VALUE_DIM, gmin, gmax);
        }
    }

    // Check input embeddings
    printf("\n=== Input Embeddings ===\n");
    float emax = -1e30f, emin = 1e30f;
    int total = model.vocab_size * D_MODEL;
    for (int i = 0; i < total && i < 100000; i++) {
        float v = model.token_embd[i];
        if (v > emax) emax = v;
        if (v < emin) emin = v;
    }
    printf("  token_embd first 100K: min=%.2f max=%.2f\n", emin, emax);

    // Check output weight
    if (model.output_weight) {
        float owmax = -1e30f, owmin = 1e30f;
        for (int i = 0; i < total && i < 100000; i++) {
            float v = model.output_weight[i];
            if (v > owmax) owmax = v;
            if (v < owmin) owmin = v;
        }
        printf("  output_weight first 100K: min=%.2f max=%.2f\n", owmin, owmax);
    } else {
        printf("  output_weight: NULL\n");
    }

    // Check layer norms
    for (int l = 0; l < 2; l++) {
        float nmax = -1e30f, nmin = 1e30f;
        for (int i = 0; i < D_MODEL; i++) {
            float v = model.layers[l].attn_norm_weight[i];
            if (v > nmax) nmax = v;
            if (v < nmin) nmin = v;
        }
        printf("  layer %d attn_norm: min=%.2f max=%.2f\n", l, nmin, nmax);

        nmax = -1e30f; nmin = 1e30f;
        for (int i = 0; i < D_MODEL; i++) {
            float v = model.layers[l].post_attn_norm_weight[i];
            if (v > nmax) nmax = v;
            if (v < nmin) nmin = v;
        }
        printf("  layer %d post_attn_norm: min=%.2f max=%.2f\n", l, nmin, nmax);
    }

    wubu_model_free(&model);
    return 0;
}
