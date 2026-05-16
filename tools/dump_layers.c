/**
 * dump_layers.c — Run 1 token through our model, dump per-layer hidden states.
 * Compares SSM vs GQA layer outputs.
 *
 * Build: gcc -O2 -I include -o dump_layers tools/dump_layers.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/qlearner.o -lm -fopenmp -lLLama
 *
 * Usage: ./dump_layers model.gguf
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declaration for GQA forward
void wubu_gqa_forward(const float *x, int B, int T, const gqa_weights *w,
                      kv_cache_t *cache, const float *rope_sc, float *output);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    wubu_model mdl;
    if (wubu_model_load(argv[1], &mdl, 0) != 0) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }

    int D = mdl.d_model;
    int token_id = 248044; // BOS
    float *embd = mdl.token_embd;
    if (!embd) { fprintf(stderr, "No embd\n"); return 1; }

    float x[D];
    memcpy(x, embd + token_id * D, D * sizeof(float));

    printf("{\n  \"model\": \"%s\",\n  \"layers\": %d,\n  \"d_model\": %d,\n",
           argv[1], mdl.n_layers, D);

    float residual[D];
    memcpy(residual, x, D * sizeof(float));
    float normed[D], attn[D], ffn[D];

    for (int l = 0; l < mdl.n_layers; l++) {
        // Pre-attention norm
        wubu_rms_norm(1, 1, D, residual, mdl.layers[l].attn_norm_weight, 1e-6f, normed);

        float emb_mean = 0, emb_max = -1e30, emb_min = 1e30;
        for (int i = 0; i < D; i++) { emb_mean += normed[i]; if (normed[i] > emb_max) emb_max = normed[i]; if (normed[i] < emb_min) emb_min = normed[i]; }
        emb_mean /= D;

        if (mdl.layers[l].is_ssm) {
            wubu_ssm_forward(normed, 1, 1, &mdl.layers[l].w.ssm,
                             mdl.layers[l].ssm_state,
                             mdl.layers[l].conv_state, attn);
        } else {
            wubu_gqa_forward(normed, 1, 1, &mdl.layers[l].w.gqa,
                             mdl.layers[l].kv_cache,
                             mdl.layers[l].rope_sc, attn);
        }

        float att_mean = 0, att_max = -1e30, att_min = 1e30;
        for (int i = 0; i < D; i++) { att_mean += attn[i]; if (attn[i] > att_max) att_max = attn[i]; if (attn[i] < att_min) att_min = attn[i]; }
        att_mean /= D;
        int has_nan = 0; for (int i = 0; i < D; i++) if (isnan(attn[i]) || isinf(attn[i])) { has_nan = 1; break; }

        printf("  \"L%d\": {\n", l);
        printf("    \"type\": \"%s\",\n", mdl.layers[l].is_ssm ? "SSM" : "GQA");
        printf("    \"input\": {\"mean\": %.6f, \"max\": %.6f, \"min\": %.6f},\n", emb_mean, emb_max, emb_min);
        printf("    \"attn\": {\"mean\": %.6f, \"max\": %.6f, \"min\": %.6f, \"nan\": %d},\n", att_mean, att_max, att_min, has_nan);
        
        // Dump first 5 attn values for comparison
        printf("    \"attn[:5]\": [%.8f, %.8f, %.8f, %.8f, %.8f],\n", attn[0], attn[1], attn[2], attn[3], attn[4]);
        printf("    \"attn[100:105]\": [%.8f, %.8f, %.8f, %.8f, %.8f]\n", attn[100], attn[101], attn[102], attn[103], attn[104]);

        // Residual add
        for (int i = 0; i < D; i++) residual[i] += attn[i];
    }

    printf("  \"_comment\": \"Run llama.cpp with same input and compare attn means.\\n  If SSM layers match but GQA don't (or vice versa), the other path has the bug.\\n  If both diverge at same layer, check earlier layer propagation.\\\"\\n}\");\n");
    printf("}\n");

    wubu_model_free(&mdl);
    return 0;
}
