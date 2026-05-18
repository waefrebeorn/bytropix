/**
 * check_router.c — Compare router computation between our code and reference.
 * Dumps router logits for layer 0 with a simple input.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    moe_weights_t moe;
    if (!wubu_moe_load_layer(ctx, 0, &moe)) return 1;

    // Input: all ones
    float x[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) x[i] = 1.0f;

    // Router: our method
    float our_scores[N_EXPERTS];
    wubu_moe_router(x, 1, 1, moe.ffn_gate_inp, our_scores);

    printf("=== Our Router (layer 0) ===\n");
    printf("First 16 scores: ");
    for (int e = 0; e < 16; e++) printf("%+.4f ", our_scores[e]);
    printf("\n");

    // Softmax
    float mx = our_scores[0];
    for (int e = 1; e < N_EXPERTS; e++) if (our_scores[e] > mx) mx = our_scores[e];
    float sum_exp = 0;
    for (int e = 0; e < N_EXPERTS; e++) sum_exp += expf(our_scores[e] - mx);
    float inv = 1.0f / sum_exp;
    float sm[N_EXPERTS];
    for (int e = 0; e < N_EXPERTS; e++) sm[e] = expf(our_scores[e] - mx) * inv;

    // Top-8
    int topk[8];
    float topw[8];
    for (int k = 0; k < 8; k++) {
        int best = -1; float best_v = -1e30f;
        for (int e = 0; e < N_EXPERTS; e++) {
            int used = 0;
            for (int p = 0; p < k; p++) if (topk[p] == e) { used = 1; break; }
            if (!used && sm[e] > best_v) { best_v = sm[e]; best = e; }
        }
        topk[k] = best;
        topw[k] = sm[best];
    }
    // Renormalize
    float sw = 0; for (int k = 0; k < 8; k++) sw += topw[k];
    for (int k = 0; k < 8; k++) topw[k] /= sw;

    printf("Our top-8 experts: ");
    for (int k = 0; k < 8; k++) printf("%d(%.4f) ", topk[k], topw[k]);
    printf("\n");

    // Now compute the router via reference (ggml mul_mat)
    // We'll use the router weights directly: scores[e] = sum_k x[k] * ffn_gate_inp[k + e * D_MODEL]
    float ref_scores[N_EXPERTS];
    for (int e = 0; e < N_EXPERTS; e++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)moe.ffn_gate_inp[k + e * D_MODEL];
        ref_scores[e] = (float)sum;
    }

    // Check if our scores match ref
    int match = 1;
    for (int e = 0; e < N_EXPERTS; e++) {
        if (fabsf(our_scores[e] - ref_scores[e]) > 1e-4f) { match = 0; break; }
    }
    printf("Router match: %s\n", match ? "YES" : "NO");
    if (!match) {
        printf("Max diff: %.10f\n", (double)fabsf(our_scores[0] - ref_scores[0]));
    }

    // Check: is the access pattern k + e * D_MODEL correct?
    // The tensor dims [D_MODEL, N_EXPERTS] = [2048, 256]
    // dims[0] = 2048 innermost, dims[1] = 256
    // offset = k + e * 2048
    // Our pattern: k + e * D_MODEL = k + e * 2048
    // Alternative: e + k * N_EXPERTS = e + k * 256
    printf("\nffn_gate_inp dims check:\n");
    printf("  D_MODEL=%d, N_EXPERTS=%d\n", D_MODEL, N_EXPERTS);
    printf("  Our access: k + e * D_MODEL = k + e * %d\n", D_MODEL);
    printf("  Alt access: e + k * N_EXPERTS = e + k * %d\n", N_EXPERTS);
    
    // Check first few values at both access patterns
    printf("\n  W[0,0] our=%.6f alt=%.6f\n", moe.ffn_gate_inp[0 + 0 * D_MODEL], moe.ffn_gate_inp[0 + 0 * N_EXPERTS]);
    printf("  W[0,1] our=%.6f alt=%.6f\n", moe.ffn_gate_inp[0 + 1 * D_MODEL], moe.ffn_gate_inp[1 + 0 * N_EXPERTS]);
    printf("  W[1,0] our=%.6f alt=%.6f\n", moe.ffn_gate_inp[1 + 0 * D_MODEL], moe.ffn_gate_inp[0 + 1 * N_EXPERTS]);

    wubu_moe_free_layer(&moe);
    gguf_close(ctx);
    return 0;
}
