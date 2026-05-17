/**
 * dump_layers_moe.c — Dump per-layer residuals WITH MoE enabled.
 * Uses wubu_model_forward (library path) with enable_moe=true.
 * Build: gcc -O2 -I include -o dump_layers_moe tools/dump_layers_moe.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o -lm -fopenmp
 * Usage: ./dump_layers_moe model.gguf
 * Output: /tmp/libmoe_l{layer}_postattn.bin, _postmoe.bin
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, argv[1])) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }
    fprintf(stderr, "Model init done, enable_moe=%d\n", mdl.enable_moe);
    // Enable MoE
    mdl.enable_moe = true;

    int D = D_MODEL;
    int L = mdl.n_layers;
    float *residual = (float *)malloc(D_MODEL * sizeof(float));

    float *embed_tensor = mdl.token_embd;
    int token_id = 248044; // BOS
    fprintf(stderr, "embed_tensor=%p use_embedding_file=%d n_layers=%d\n", (void*)embed_tensor, mdl.use_embedding_file, mdl.n_layers);
    fflush(stderr);
    float *x = embed_tensor;
    if (x) x += token_id * D_MODEL;
    if (!x) {
        fprintf(stderr, "Error: no token_embd available\n");
        wubu_model_free(&mdl);
        return 1;
    }
    memcpy(residual, x, D_MODEL * sizeof(float));
    fprintf(stderr, "Starting forward loop: L=%d\n", L);
    fflush(stderr);

    for (int l = 0; l < L && l < 40; l++) {
        // Pre-attention RMSNorm
        float normed[D_MODEL];
        wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].attn_norm_weight, 1e-6f, normed);

        float attn[D_MODEL];
        memset(attn, 0, D_MODEL * sizeof(float));

        if (mdl.layers[l].is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &mdl.layers[l].ssm, ssm_state, conv_state, attn);
        } else {
            wubu_gqa_forward(normed, 1, 1, &mdl.layers[l].gqa, attn);
        }

        // Residual = x + attn_out
        for (int i = 0; i < D_MODEL; i++) residual[i] += attn[i];

        // Post-attention RMSNorm
        float normed2[D_MODEL];
        wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].post_attn_norm_weight, 1e-6f, normed2);

        fprintf(stderr, "  L%d: loading MoE...\n", l);
        fflush(stderr);
        moe_weights_t moe_w;
        memset(&moe_w, 0, sizeof(moe_w));
        int loaded = wubu_moe_load_layer(mdl.gguf_ctx, l, &moe_w);
        fprintf(stderr, "  L%d: MoE loaded=%d\n", l, loaded);
        fflush(stderr);
        if (loaded) {
            float moe_out[D_MODEL];
            wubu_moe_forward(normed2, 1, 1, &moe_w, moe_out);
            wubu_moe_free_layer(&moe_w);
            for (int i = 0; i < D_MODEL; i++) residual[i] += moe_out[i];
        } else {
            // Fallback: pass-through
            for (int i = 0; i < D_MODEL; i++) residual[i] += normed2[i];
        }

        // Dump
        char fn[256];
        snprintf(fn, sizeof(fn), "/tmp/libmoe_l%02d_residual.bin", l);
        FILE *f = fopen(fn, "wb");
        if (f) { fwrite(residual, sizeof(float), D_MODEL, f); fclose(f); }

        double s = 0; for (int i = 0; i < D_MODEL; i++) s += (double)residual[i] * residual[i];
        printf("L%02d: residual rms=%.4f\n", l, sqrtf(s / D_MODEL));
    }

    free(residual);
    wubu_model_free(&mdl);
    printf("Done.\n");
    return 0;
}
