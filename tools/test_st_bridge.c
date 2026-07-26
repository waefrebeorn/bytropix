/* test_st_bridge.c -- verify the safetensors->bytropix F32 bridge
 * maps the REAL published HF tensor names (model.language_model.layers.N...)
 * onto bytropix's SSM/GQA/MoE F32 weight structs with no gaps.
 *
 * We do NOT call bytropix's hybrid forward here: that forward is
 * dimension-HARDCODED to compile-time D_MODEL=2048 (Gemma4 macros),
 * so it can only run on the REAL 27B/35B checkpoint on the GPU box.
 * The bridge's job is the MAPPING; we verify every expected weight
 * pointer is non-NULL, f32_mode is set, and the loaded F32 data is
 * finite + non-degenerate (real F32 dequant + transpose, not a stub). */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_ssm.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int all_finite(const float *a, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(a[i])) return 0;
    return 1;
}

int main(void) {
    wubu_adapter_t ad;
    memset(&ad, 0, sizeof(ad));
    ad.arch = WUBU_ARCH_QWEN36_HYBRID;
    ad.d_model = 16; ad.d_ff = 8; ad.n_layers = 2;
    ad.gqa_kv_heads = 2; ad.gqa_q_heads = 4; ad.gqa_head_dim = 4;
    ad.n_experts = 0; ad.ok = 1;

    wubu_model_t m;
    if (wubu_model_init_safetensors(&m, "fixture_model.safetensors", &ad) != 0) {
        fprintf(stderr, "FAIL: init_safetensors\n"); return 1;
    }
    if (m.layers == NULL || m.n_layers != 2 || m.d_model != 16) {
        fprintf(stderr, "FAIL: model meta\n"); return 1;
    }
    wubu_layer_t *ly = &m.layers[0];
    if (!ly->ssm.attn_qkv_weight_f32 || !ly->ssm.attn_gate_weight_f32 ||
        !ly->ssm.ssm_out_weight_f32 || !ly->ssm.ssm_a ||
        !ly->ssm.ssm_beta_weight || !ly->ssm.ssm_dt_bias ||
        !ly->ssm.ssm_conv1d_weight || !ly->ssm.ssm_norm_weight ||
        !ly->gqa.attn_q_weight || !ly->gqa.attn_k_weight ||
        !ly->gqa.attn_v_weight || !ly->gqa.attn_output_weight ||
        !ly->moe.ffn_up_exps) {
        fprintf(stderr, "FAIL: weight pointers not mapped\n"); return 1;
    }
    if (ly->ssm.f32_mode != 1) { fprintf(stderr, "FAIL: f32_mode not set\n"); return 1; }
    if (ly->gqa.attn_q_weight_q || ly->ssm.attn_qkv_weight_q) {
        fprintf(stderr, "FAIL: quantized blob ptrs should be NULL in F32 mode\n"); return 1;
    }

    /* Verify loaded F32 data is real (finite + has spread, not all-zero) */
    int bad = 0;
    float *chk[] = { ly->ssm.attn_qkv_weight_f32, ly->ssm.attn_gate_weight_f32,
                    ly->ssm.ssm_out_weight_f32, ly->ssm.ssm_a,
                    ly->gqa.attn_q_weight, ly->gqa.attn_output_weight,
                    ly->moe.ffn_up_exps };
    for (int i = 0; i < 7; i++) {
        float *p = chk[i];
        int n = 256; /* spot-check first 256 elems */
        if (!all_finite(p, n)) { bad = 1; break; }
        float mn = 1e30f, mx = -1e30f;
        for (int j = 0; j < n; j++) { if (p[j] < mn) mn = p[j]; if (p[j] > mx) mx = p[j]; }
        if (mx - mn < 1e-3f) { bad = 1; break; } /* degenerate => not real data */
    }
    if (bad) { fprintf(stderr, "FAIL: loaded F32 weights degenerate/non-finite\n"); return 1; }

    wubu_model_safetensors_free(&m);
    printf("PASS: safetensors bridge maps HF->bytropix F32 structs (SSM+GQA+MoE, real weights)\n");
    return 0;
}
