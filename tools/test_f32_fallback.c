// F32 fallback test: measure architecture-only error vs quantized error
// Compares: our F32 path vs reference, our quantized path vs reference
// This isolates architecture errors from quantization noise.
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    // Load model normally (quantized)
    wubu_model_t mdl_q;
    if (!wubu_model_init(&mdl_q, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl_q.enable_moe = true;

    // Load a second copy for F32 fallback
    wubu_model_t mdl_f32;
    if (!wubu_model_init(&mdl_f32, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl_f32.enable_moe = true;

    // Clear ALL quantized pointers in mdl_f32 to force F32 fallback
    for (int l = 0; l < mdl_f32.n_layers; l++) {
        wubu_layer_t *layer = &mdl_f32.layers[l];
        if (layer->is_ssm) {
            layer->ssm.attn_qkv_weight_q = NULL;
            layer->ssm.attn_gate_weight_q = NULL;
            layer->ssm.ssm_out_weight_q = NULL;
        } else {
            layer->gqa.attn_q_weight_q = NULL;
            layer->gqa.attn_k_weight_q = NULL;
            layer->gqa.attn_v_weight_q = NULL;
            layer->gqa.attn_output_weight_q = NULL;
        }
        // MoE quantized
        moe_weights_t *moe = &layer->moe;
        moe->ffn_gate_exps_q = NULL;
        moe->ffn_gate_exps_q_type = GGML_TYPE_F32;
        moe->ffn_up_exps_q = NULL;
        moe->ffn_up_exps_q_type = GGML_TYPE_F32;
        moe->ffn_down_exps_q = NULL;
        moe->ffn_down_exps_q_type = GGML_TYPE_F32;
        moe->ffn_gate_shexp_q = NULL;
        moe->ffn_gate_shexp_q_type = GGML_TYPE_F32;
        moe->ffn_up_shexp_q = NULL;
        moe->ffn_up_shexp_q_type = GGML_TYPE_F32;
        moe->ffn_down_shexp_q = NULL;
        moe->ffn_down_shexp_q_type = GGML_TYPE_F32;
    }
    // Clear output weight quantized
    mdl_f32.output_weight_q = NULL;
    mdl_f32.output_weight_type = GGML_TYPE_F32;

    int D = D_MODEL;
    int vs = mdl_q.vocab_size;

    // Load same embedding for both
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl_q.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR: can't open emb file\n"); return 1; }
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl_q.token_embd + 248044LL * D, D * sizeof(float));
    }

    // Run F32 version
    float *logits_f32 = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl_f32, x, 1, 1, logits_f32);

    // Run quantized version
    float *logits_q = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl_q, x, 1, 1, logits_q);

    // Load reference
    float *ref = (float *)malloc(vs * sizeof(float));
    FILE *f = fopen("/tmp/llama_logits_new.bin", "rb");
    if (!f) { printf("No reference file\n"); return 1; }
    fread(ref, sizeof(float), vs, f);
    fclose(f);

    // Compare F32 vs ref
    double dot_f32 = 0, n_f32 = 0, n_ref = 0;
    double dot_q = 0, n_q = 0;
    double max_diff_f32 = 0;
    for (int i = 0; i < vs; i++) {
        double d_f32 = (double)logits_f32[i] - (double)ref[i];
        if (fabs(d_f32) > max_diff_f32) max_diff_f32 = fabs(d_f32);
        dot_f32 += (double)logits_f32[i] * (double)ref[i];
        n_f32 += (double)logits_f32[i] * (double)logits_f32[i];
        n_ref += (double)ref[i] * (double)ref[i];
        dot_q += (double)logits_q[i] * (double)ref[i];
        n_q += (double)logits_q[i] * (double)logits_q[i];
    }
    double cos_f32 = dot_f32 / (sqrt(n_f32) * sqrt(n_ref));
    double cos_q = dot_q / (sqrt(n_q) * sqrt(n_ref));

    printf("========================================\n");
    printf("F32 FALLBACK TEST\n");
    printf("========================================\n");
    printf("F32 cos-sim vs ref:    %.10f\n", cos_f32);
    printf("QNT cos-sim vs ref:    %.10f (baseline = 0.9968)\n", cos_q);
    printf("F32 max diff vs ref:   %.10f\n", max_diff_f32);
    printf("Quantization error:    %.10f (F32 - quantized)\n", cos_f32 - cos_q);
    printf("Architecture error:    %.10f (F32 - 1.0 = architecture gaps)\n", 1.0 - cos_f32);
    printf("\nTop-5 F32 logits:\n");
    {
        float *cpy = (float *)malloc(vs * sizeof(float));
        memcpy(cpy, logits_f32, vs * sizeof(float));
        for (int k = 0; k < 5; k++) {
            float best = -1e30f; int best_idx = -1;
            for (int i = 0; i < vs; i++) if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
            cpy[best_idx] = -1e30f;
            printf("  [%d] val=%.10f\n", best_idx, (double)best);
        }
        free(cpy);
    }
    printf("\nTop-5 REF logits:\n");
    {
        float *cpy = (float *)malloc(vs * sizeof(float));
        memcpy(cpy, ref, vs * sizeof(float));
        for (int k = 0; k < 5; k++) {
            float best = -1e30f; int best_idx = -1;
            for (int i = 0; i < vs; i++) if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
            cpy[best_idx] = -1e30f;
            printf("  [%d] val=%.10f\n", best_idx, (double)best);
        }
        free(cpy);
    }

    free(ref);
    free(logits_q);
    free(logits_f32);
    free(x);
    wubu_model_free(&mdl_f32);
    wubu_model_free(&mdl_q);
    return 0;
}
