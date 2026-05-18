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

            int n_out = VALUE_DIM * D_MODEL;
            float omax = -1e30f, omin = 1e30f;
            for (int i = 0; i < n_out; i++) {
                float v = model.layers[l].ssm.ssm_out_weight[i];
                if (v > omax) omax = v;
                if (v < omin) omin = v;
            }
            printf("  ssm_out [%dx%d]: min=%.2f max=%.2f\n", VALUE_DIM, D_MODEL, omin, omax);

            float nmax = -1e30f, nmin = 1e30f;
            for (int i = 0; i < SSM_D_STATE; i++) {
                float v = model.layers[l].ssm.ssm_norm_weight[i];
                if (v > nmax) nmax = v;
                if (v < nmin) nmin = v;
            }
            printf("  ssm_norm [%d]: min=%.2f max=%.2f\n", SSM_D_STATE, nmin, nmax);
        }
    }

    // Check layer norms first few values
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
