/**
 * tools/dump_moe_layer0.c
 *
 * Load Layer 0 of the model WITH MoE and dump ALL MoE intermediates.
 * Single token (B=1, T=1).
 *
 * Dumps:
 *   - /tmp/dbg_moe_input.bin     — MoE input (post-attention norm)
 *   - /tmp/dbg_moe_scores.bin    — router scores [256] (pre-softmax)
 *   - /tmp/dbg_moe_probs.bin     — softmax probs [256]
 *   - /tmp/dbg_moe_topk_idx.bin  — top-8 expert indices [8]
 *   - /tmp/dbg_moe_topk_wgt.bin  — top-8 expert weights [8] (normalized)
 *   - /tmp/dbg_moe_routed.bin    — routed expert output [2048]
 *   - /tmp/dbg_moe_shared.bin    — shared expert output [2048] (before gating)
 *   - /tmp/dbg_moe_shared_gate.bin — shared expert gate scalar
 *   - /tmp/dbg_moe_output.bin    — final MoE output [2048]
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void dump_bin(const char *path, const float *data, int n) {
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    int D = D_MODEL;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR: can't open emb\n"); return 1; }
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f); fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Layer 0
    wubu_layer_t *layer = &mdl.layers[0];
    
    // Pre-attention norm
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, layer->attn_norm_weight, 1e-6f, normed);
    
    // SSM forward
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *attn_out = (float *)malloc(D * sizeof(float));
    wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out);
    
    // Residual
    for (int i = 0; i < D; i++) x[i] += attn_out[i];
    
    // Post-attention norm
    float *normed2 = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, layer->post_attn_norm_weight, 1e-6f, normed2);
    dump_bin("/tmp/dbg_moe_input.bin", normed2, D);
    
    // Load MoE weights (F32)
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) {
        printf("ERROR: failed to load MoE layer 0\n"); return 1;
    }
    
    // Route: compute scores
    float scores[N_EXPERTS];
    wubu_moe_router(normed2, 1, 1, moe.ffn_gate_inp, scores);
    dump_bin("/tmp/dbg_moe_scores.bin", scores, N_EXPERTS);
    printf("Router scores[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.6f", scores[i]);
    printf("\n");
    
    // Compute softmax + top-k + normalize (same as wubu_moe_forward)
    float max_s = scores[0];
    for (int e = 1; e < N_EXPERTS; e++) if (scores[e] > max_s) max_s = scores[e];
    float sum_exp = 0;
    for (int e = 0; e < N_EXPERTS; e++) sum_exp += expf(scores[e] - max_s);
    float inv_sum = 1.0f / (sum_exp + 1e-30f);
    float probs[N_EXPERTS];
    for (int e = 0; e < N_EXPERTS; e++) probs[e] = expf(scores[e] - max_s) * inv_sum;
    dump_bin("/tmp/dbg_moe_probs.bin", probs, N_EXPERTS);
    
    // Top-8
    int indices[N_ACTIVE_EXPTS];
    float weights[N_ACTIVE_EXPTS];
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
        indices[k] = k;
        weights[k] = probs[k];
    }
    for (int i = 0; i < N_ACTIVE_EXPTS-1; i++)
        for (int j = i+1; j < N_ACTIVE_EXPTS; j++)
            if (weights[i] > weights[j]) {
                float tw = weights[i]; weights[i] = weights[j]; weights[j] = tw;
                int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
            }
    for (int e = N_ACTIVE_EXPTS; e < N_EXPERTS; e++) {
        if (probs[e] > weights[0]) {
            weights[0] = probs[e];
            indices[0] = e;
            int pos = 0;
            while (pos + 1 < N_ACTIVE_EXPTS && weights[pos] > weights[pos+1]) {
                float tw = weights[pos]; weights[pos] = weights[pos+1]; weights[pos+1] = tw;
                int ti = indices[pos]; indices[pos] = indices[pos+1]; indices[pos+1] = ti;
                pos++;
            }
        }
    }
    
    // Normalize
    float sum_w = 0;
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += weights[k];
    if (sum_w > 1e-30f) {
        float inv_sum_w = 1.0f / sum_w;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) weights[k] *= inv_sum_w;
    }
    
    dump_bin("/tmp/dbg_moe_topk_idx.bin", (float*)indices, N_ACTIVE_EXPTS);
    dump_bin("/tmp/dbg_moe_topk_wgt.bin", weights, N_ACTIVE_EXPTS);
    
    printf("Top-8 experts:");
    for (int k = 0; k < N_ACTIVE_EXPTS; k++)
        printf(" [%d] w=%.6f", indices[k], weights[k]);
    printf("\n");
    
    // Compute full MoE output (shared expert + routed)
    float *ffn_out = (float *)malloc(D * sizeof(float));
    memcpy(ffn_out, normed2, D * sizeof(float)); // passthrough placeholder
    
    // Actually call the forward function
    wubu_moe_forward(normed2, 1, 1, &moe, ffn_out, NULL);
    dump_bin("/tmp/dbg_moe_output.bin", ffn_out, D);
    
    printf("MoE output[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
           ffn_out[0], ffn_out[1], ffn_out[2], ffn_out[3], ffn_out[4]);
    {
        float m=0,s=0;
        for (int i=0;i<D;i++){m+=ffn_out[i];s+=ffn_out[i]*ffn_out[i];}
        printf("MoE out: mean=%.6f std=%.6f\n", m/D, sqrtf(s/D - (m/D)*(m/D)));
    }
    
    wubu_moe_free_layer(&moe);
    free(x); free(normed); free(ssm_state); free(conv_state);
    free(attn_out); free(normed2); free(ffn_out);
    wubu_model_free(&mdl);
    return 0;
}
