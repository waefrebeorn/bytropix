// Quick test: examine ssm_a values
#include "wubu_model.h"
#include <stdio.h>
#include <math.h>

int main() {
    wubu_model_t model;
    if (!wubu_model_init(&model, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) {
        fprintf(stderr, "FAIL load model\n");
        return 1;
    }
    printf("n_layers=%d DT_RANK=%d\n", model.n_layers, DT_RANK);
    for (int l = 0; l < model.n_layers && l < 5; l++) {
        if (model.layers[l].is_ssm) {
            printf("\n--- Layer %d SSM ssm_a ---\n", l);
            float min = 1e30f, max = -1e30f;
            int npos = 0, nneg = 0, nzero = 0;
            for (int j = 0; j < DT_RANK; j++) {
                float v = model.layers[l].ssm.ssm_a[j];
                if (v > max) max = v;
                if (v < min) min = v;
                if (v > 0) npos++;
                else if (v < 0) nneg++;
                else nzero++;
            }
            printf("  min=%f max=%f pos=%d neg=%d zero=%d\n", min, max, npos, nneg, nzero);
            printf("  first 8: ");
            for (int j = 0; j < 8 && j < DT_RANK; j++)
                printf("%f ", model.layers[l].ssm.ssm_a[j]);
            printf("\n");

            printf("  dt_bias first 8: ");
            for (int j = 0; j < 8 && j < DT_RANK; j++)
                printf("%f ", model.layers[l].ssm.ssm_dt_bias[j]);
            printf("\n");
        }
    }

    int l = 0;
    if (model.layers[l].is_ssm) {
        printf("\n--- Layer %d ssm_alpha_weight sample (first 5 x 5) ---\n", l);
        for (int i = 0; i < 5 && i < 10; i++) {
            for (int j = 0; j < 5 && j < DT_RANK; j++)
                printf("%f ", model.layers[l].ssm.ssm_alpha_weight[i * DT_RANK + j]);
            printf("\n");
        }

        printf("\n--- Simulated gate values for random input (layer %d) ---\n", l);
        float input[10] = {0.1f, -0.2f, 0.3f, -0.1f, 0.05f, 0.0f, -0.15f, 0.25f, -0.05f, 0.12f};
        for (int n = 0; n < 3; n++) {
            float alpha_val[DT_RANK];
            for (int j = 0; j < DT_RANK; j++) {
                double sum = 0;
                for (int i = 0; i < 10; i++)
                    sum += (double)input[i] * (double)model.layers[l].ssm.ssm_alpha_weight[i * DT_RANK + j];
                alpha_val[j] = (float)sum;
            }
            printf("  Input sample %d:\n", n);
            for (int j = 0; j < 4 && j < DT_RANK; j++) {
                float sp = alpha_val[j] + model.layers[l].ssm.ssm_dt_bias[j];
                float softp = sp > 80.0f ? 80.0f : sp < -80.0f ? 0.0f : logf(1.0f + expf(sp));
                float gate = softp * model.layers[l].ssm.ssm_a[j];
                float egate = expf(gate);
                printf("    rank[%d]: alpha_raw=%.4f + dt_bias=%.4f = %.4f -> softplus=%.4f * ssm_a=%.4f = gate=%.4f -> exp(gate)=%.4f\n",
                    j, alpha_val[j], model.layers[l].ssm.ssm_dt_bias[j],
                    alpha_val[j] + model.layers[l].ssm.ssm_dt_bias[j],
                    softp, model.layers[l].ssm.ssm_a[j], gate, egate);
            }
            for (int i = 0; i < 10; i++) input[i] += 0.1f;
        }
    }

    wubu_model_free(&model);
    return 0;
}
