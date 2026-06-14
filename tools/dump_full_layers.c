/**
 * dump_full_layers.c — Dump per-layer residuals for comparison with llama.cpp.
 * Build: gcc -O2 -I include -o dump_full_layers tools/dump_full_layers.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o -lm -fopenmp
 * Usage: ./dump_full_layers model.gguf
 * Output: /tmp/bytropix_l{layer}_residual.bin (after attention + FFN pass-through)
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

    int D = D_MODEL;  // 2048
    int L = mdl.n_layers;
    float *residual = (float *)malloc(D_MODEL * sizeof(float));

    float *embed_tensor = mdl.token_embd;
    int token_id = 248044; // BOS
    float *x = NULL;
    
    if (mdl.use_embedding_file) {
        // Load token 0 embedding from file
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (!emb_f) { fprintf(stderr, "Failed to open %s\n", emb_path); return 1; }
        float *all_emb = (float *)malloc((int64_t)mdl.vocab_size * D_MODEL * sizeof(float));
        fread(all_emb, sizeof(float), (int64_t)mdl.vocab_size * D_MODEL, emb_f);
        fclose(emb_f);
        x = all_emb + token_id * D_MODEL;
        printf("Loaded embeddings from file\n");
    } else {
        x = embed_tensor + token_id * D_MODEL;
        printf("Using in-memory embeddings\n");
    }
    // Copy embedding as initial residual
    memcpy(residual, x, D_MODEL * sizeof(float));

    // Dump initial residual (embedding only)
    FILE *f = fopen("/tmp/bytropix_embed.bin", "wb");
    if (f) { fwrite(residual, sizeof(float), D, f); fclose(f); }

    for (int l = 0; l < L; l++) {
        // Pre-attention RMSNorm
        float normed[D_MODEL];
        wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].attn_norm_weight, 1e-6f, normed);

        float attn[D_MODEL];
        memset(attn, 0, D_MODEL * sizeof(float));

        if (mdl.layers[l].is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &mdl.layers[l].ssm, ssm_state, conv_state, attn, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, 1, 1, &mdl.layers[l].gqa, attn, NULL, NULL, 0, NULL, NULL, mdl.layers[l].gqa.head_dim, mdl.layers[l].gqa.q_heads, mdl.layers[l].gqa.kv_heads);
        }

        // Dump attention output (before residual)
        char fn[256];
        snprintf(fn, sizeof(fn), "/tmp/bytropix_l%02d_attn.bin", l);
        f = fopen(fn, "wb");
        if (f) { fwrite(attn, sizeof(float), D_MODEL, f); fclose(f); }

        // Residual = x + attn_out
        for (int i = 0; i < D_MODEL; i++) residual[i] += attn[i];

        // Post-attention RMSNorm
        float normed2[D_MODEL];
        wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].post_attn_norm_weight, 1e-6f, normed2);
        // No MoE - pass through (skip FFN entirely)
        for (int i = 0; i < D_MODEL; i++) residual[i] += normed2[i];

        // Dump residual after layer l (matches llama's post_moe dump DIRECTLY)
        snprintf(fn, sizeof(fn), "/tmp/bytropix_l%02d_residual.bin", l);
        f = fopen(fn, "wb");
        if (f) { fwrite(residual, sizeof(float), D_MODEL, f); fclose(f); }

        if (l < 3 || l == L-1)
            printf("L%02d: residual rms=%.4f\n", l, sqrtf(1.0f/D_MODEL * (
                double)((long)residual[0]*(long)residual[0] + (long)residual[D_MODEL-1]*(long)residual[D_MODEL-1])));
    }

    // Final RMSNorm
    float final_normed[D_MODEL];
    wubu_rms_norm(1, 1, D_MODEL, residual, mdl.norm_weight, 1e-6f, final_normed);

    f = fopen("/tmp/bytropix_final_hidden.bin", "wb");
    if (f) { fwrite(final_normed, sizeof(float), D_MODEL, f); fclose(f); }

    free(residual);
    wubu_model_free(&mdl);
    printf("Done. Check /tmp/bytropix_*.bin\n");
    return 0;
}
