/**
 * test_decode_path.c — Full decode path verification
 *
 * Compares prefill(T=7, token 6) vs prefill(T=6) + decode(T=1) with SSM state carry.
 * Per-layer dump to pinpoint first divergence (SSM layers should match, GQA layers won't
 * without KV cache).
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;

    const int B = 1;
    const int T_pre = 6;    // prefill tokens
    const int T_full = 7;   // prefill+1 tokens for reference
    const int N_pre = B * T_pre;      // 6
    const int N_full = B * T_full;    // 7

    printf("\n=== Full Decode Path Test ===\n");
    printf("Model: %d layers (%d SSM, %d GQA)\n",
           mdl.n_layers,
           mdl.n_layers - mdl.n_layers/4,
           mdl.n_layers/4);

    // ===== Generate random embeddings =====
    float *embd = (float *)malloc(N_full * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N_full * D_MODEL; i++)
        embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    printf("\nEmbeddings generated: T_full=%d, T_pre=%d\n", T_full, T_pre);

    // ===== Per-layer storage =====
    // ref_hidden[l][d] = residual stream at position T_pre (token 6) after layer l, T=7 prefill
    // dec_hidden[l][d] = residual stream after layer l, decode(T=1) after prefill(T=6)
    float *ref_hidden = (float *)malloc(mdl.n_layers * D_MODEL * sizeof(float));
    float *dec_hidden = (float *)malloc(mdl.n_layers * D_MODEL * sizeof(float));
    memset(ref_hidden, 0, mdl.n_layers * D_MODEL * sizeof(float));
    memset(dec_hidden, 0, mdl.n_layers * D_MODEL * sizeof(float));

    // ============================================================
    // PASS 1: Prefill T=7 (reference) — capture hidden[6] per layer
    // ============================================================
    printf("\n--- PASS 1: Prefill T=%d ---\n", T_full);

    // Reset SSM states to zero
    memset(mdl.ssm_states, 0,
           mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    memset(mdl.conv_states, 0,
           mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));

    float *x_ref = (float *)malloc(N_full * D_MODEL * sizeof(float));
    memcpy(x_ref, embd, N_full * D_MODEL * sizeof(float));

    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];

        // Pre-attention RMSNorm
        float *normed = (float *)malloc(N_full * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T_full, D_MODEL, x_ref, layer->attn_norm_weight, 1e-6f, normed);

        // Attention
        float *attn_out = (float *)malloc(N_full * D_MODEL * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T_full, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, B, T_full, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }

        // Residual: x = x + attn_out
        for (int i = 0; i < N_full * D_MODEL; i++)
            x_ref[i] += attn_out[i];

        // Capture hidden state at position T_pre (0-indexed: 6th token, the last one)
        memcpy(ref_hidden + l * D_MODEL, x_ref + T_pre * D_MODEL, D_MODEL * sizeof(float));

        // Post-attention RMSNorm + MoE passthrough
        float *normed2 = (float *)malloc(N_full * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T_full, D_MODEL, x_ref, layer->post_attn_norm_weight, 1e-6f, normed2);

        float *ffn_out = (float *)malloc(N_full * D_MODEL * sizeof(float));
        wubu_moe_forward(normed2, B, T_full, &layer->moe, ffn_out, NULL);

        // Residual: x = x + ffn_out
        for (int i = 0; i < N_full * D_MODEL; i++)
            x_ref[i] += ffn_out[i];

        free(normed);
        free(normed2);
        free(attn_out);
        free(ffn_out);
    }

    // Final RMSNorm
    float *ref_final = (float *)malloc(D_MODEL * sizeof(float));
    if (mdl.norm_weight) {
        float *final_normed = (float *)malloc(N_full * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T_full, D_MODEL, x_ref, mdl.norm_weight, 1e-6f, final_normed);
        memcpy(ref_final, final_normed + T_pre * D_MODEL, D_MODEL * sizeof(float));
        free(final_normed);
    } else {
        memcpy(ref_final, x_ref + T_pre * D_MODEL, D_MODEL * sizeof(float));
    }

    printf("PASS 1 complete.\n");

    // ============================================================
    // PASS 2: Prefill T=6 + Decode T=1 — capture decode per layer
    // ============================================================
    printf("\n--- PASS 2: Prefill T=%d + Decode T=1 ---\n", T_pre);

    // Reset SSM states to zero
    memset(mdl.ssm_states, 0,
           mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    memset(mdl.conv_states, 0,
           mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));

    // Stage 2a: Prefill T=6
    float *x_dec = (float *)malloc(N_pre * D_MODEL * sizeof(float));
    memcpy(x_dec, embd, N_pre * D_MODEL * sizeof(float));

    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];

        float *normed = (float *)malloc(N_pre * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T_pre, D_MODEL, x_dec, layer->attn_norm_weight, 1e-6f, normed);

        float *attn_out = (float *)malloc(N_pre * D_MODEL * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T_pre, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, B, T_pre, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }

        for (int i = 0; i < N_pre * D_MODEL; i++)
            x_dec[i] += attn_out[i];

        float *normed2 = (float *)malloc(N_pre * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T_pre, D_MODEL, x_dec, layer->post_attn_norm_weight, 1e-6f, normed2);

        float *ffn_out = (float *)malloc(N_pre * D_MODEL * sizeof(float));
        wubu_moe_forward(normed2, B, T_pre, &layer->moe, ffn_out, NULL);

        for (int i = 0; i < N_pre * D_MODEL; i++)
            x_dec[i] += ffn_out[i];

        free(normed);
        free(normed2);
        free(attn_out);
        free(ffn_out);
    }

    printf("Prefill T=%d complete. SSM states carried forward.\n", T_pre);

    // Stage 2b: Decode T=1 (the 7th token)
    const float *embd_decode = embd + T_pre * D_MODEL;
    float *x_decode = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(x_decode, embd_decode, D_MODEL * sizeof(float));

    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];

        // Pre-attention RMSNorm (B=1, T=1)
        float *normed = (float *)malloc(D_MODEL * sizeof(float));
        wubu_rms_norm(1, 1, D_MODEL, x_decode, layer->attn_norm_weight, 1e-6f, normed);

        // Attention
        float *attn_out = (float *)malloc(D_MODEL * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }

        // Residual
        for (int i = 0; i < D_MODEL; i++)
            x_decode[i] += attn_out[i];

        // Capture hidden state at decode step for this layer
        memcpy(dec_hidden + l * D_MODEL, x_decode, D_MODEL * sizeof(float));

        // Post-attention RMSNorm + FFN
        float *normed2 = (float *)malloc(D_MODEL * sizeof(float));
        wubu_rms_norm(1, 1, D_MODEL, x_decode, layer->post_attn_norm_weight, 1e-6f, normed2);

        float *ffn_out = (float *)malloc(D_MODEL * sizeof(float));
        wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out, NULL);

        // Residual
        for (int i = 0; i < D_MODEL; i++)
            x_decode[i] += ffn_out[i];

        free(normed);
        free(normed2);
        free(attn_out);
        free(ffn_out);
    }

    // Final RMSNorm on decode output
    float *dec_final = (float *)malloc(D_MODEL * sizeof(float));
    if (mdl.norm_weight) {
        float *final_normed = (float *)malloc(D_MODEL * sizeof(float));
        wubu_rms_norm(1, 1, D_MODEL, x_decode, mdl.norm_weight, 1e-6f, final_normed);
        memcpy(dec_final, final_normed, D_MODEL * sizeof(float));
        free(final_normed);
    } else {
        memcpy(dec_final, x_decode, D_MODEL * sizeof(float));
    }

    printf("Decode T=1 complete.\n");

    // ============================================================
    // Comparison: per-layer hidden state
    // ============================================================
    printf("\n========================================\n");
    printf(" Per-Layer Hidden State Comparison\n");
    printf("========================================\n");
    printf("Layer  Type     max_diff      mean_diff    match?\n");
    printf("------ -------- ------------ ------------ -------\n");

    int first_divergent = -1;
    for (int l = 0; l < mdl.n_layers; l++) {
        const float *ref = ref_hidden + l * D_MODEL;
        const float *dec = dec_hidden + l * D_MODEL;

        float max_diff = 0, sum_diff = 0;
        for (int i = 0; i < D_MODEL; i++) {
            float d = fabsf(ref[i] - dec[i]);
            sum_diff += d;
            if (d > max_diff) max_diff = d;
        }
        float mean_diff = sum_diff / D_MODEL;
        const char *type = mdl.layers[l].is_ssm ? "SSM" : "GQA";
        const char *match = (max_diff < 1e-5f) ? "MATCH" : "DIVERGE";

        printf("L%02d   %-8s %12.8f %12.8f %s\n",
               l, type, max_diff, mean_diff, match);

        if (first_divergent < 0 && max_diff >= 1e-5f) {
            first_divergent = l;
            // Print first few divergent samples
            printf("       First divergent values (ref vs dec):\n");
            int printed = 0;
            for (int i = 0; i < D_MODEL && printed < 5; i++) {
                float d = fabsf(ref[i] - dec[i]);
                if (d >= 1e-5f) {
                    printf("       [%d] ref=%.8f dec=%.8f diff=%.8f\n",
                           i, ref[i], dec[i], d);
                    printed++;
                }
            }
        }
    }

    // ============================================================
    // Comparison: final hidden state (after final norm, before output proj)
    // ============================================================
    printf("\n========================================\n");
    printf(" Final Hidden State Comparison\n");
    printf("========================================\n");
    float final_max_diff = 0, final_sum_diff = 0;
    int final_max_idx = 0;
    for (int i = 0; i < D_MODEL; i++) {
        float d = fabsf(ref_final[i] - dec_final[i]);
        final_sum_diff += d;
        if (d > final_max_diff) { final_max_diff = d; final_max_idx = i; }
    }
    printf("max_diff=%.8f (idx=%d)\n", final_max_diff, final_max_idx);
    printf("mean_diff=%.8f\n", final_sum_diff / D_MODEL);
    printf("ref[0..4]=%.6f %.6f %.6f %.6f %.6f\n",
           ref_final[0], ref_final[1], ref_final[2], ref_final[3], ref_final[4]);
    printf("dec[0..4]=%.6f %.6f %.6f %.6f %.6f\n",
           dec_final[0], dec_final[1], dec_final[2], dec_final[3], dec_final[4]);

    // ============================================================
    // Summary
    // ============================================================
    printf("\n========================================\n");
    printf(" ANALYSIS\n");
    printf("========================================\n");
    if (first_divergent >= 0) {
        const char *ft = mdl.layers[first_divergent].is_ssm ? "SSM" : "GQA";
        printf("First divergence at layer %d (%s)\n", first_divergent, ft);
        printf("This indicates the %s layer lacks SSM state carry or KV cache.\n",
               mdl.layers[first_divergent].is_ssm ? "SSM" : "GQA");

        // Count matches vs diverges
        int ssm_match = 0, ssm_diverge = 0, gqa_match = 0, gqa_diverge = 0;
        for (int l = 0; l < mdl.n_layers; l++) {
            float max_diff = 0;
            const float *ref = ref_hidden + l * D_MODEL;
            const float *dec = dec_hidden + l * D_MODEL;
            for (int i = 0; i < D_MODEL; i++) {
                float d = fabsf(ref[i] - dec[i]);
                if (d > max_diff) max_diff = d;
            }
            if (mdl.layers[l].is_ssm) {
                if (max_diff < 1e-5f) ssm_match++; else ssm_diverge++;
            } else {
                if (max_diff < 1e-5f) gqa_match++; else gqa_diverge++;
            }
        }
        printf("SSM layers: %d match, %d diverge\n", ssm_match, ssm_diverge);
        printf("GQA layers: %d match, %d diverge\n", gqa_match, gqa_diverge);
    } else {
        printf("All layers match! Decode path is bit-exact with prefill.\n");
    }

    // ============================================================
    // Cleanup
    // ============================================================
    free(embd);
    free(x_ref);
    free(x_dec);
    free(x_decode);
    free(ref_hidden);
    free(dec_hidden);
    free(ref_final);
    free(dec_final);

    wubu_model_free(&mdl);
    return 0;
}
