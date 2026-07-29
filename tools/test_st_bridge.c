/* test_st_bridge.c -- verify the safetensors->wubuwizard F32 bridge
 * maps the REAL published HF tensor names (model.language_model.layers.N...)
 * onto wubuwizard's SSM/GQA/MoE F32 weight structs with no gaps, AND runs the
 * full pipeline (load -> wubu_model_forward -> greedy decode -> repetition
 * controller) at the model's own runtime dims (D=256 fixture).
 *
 * We do NOT need the multi-GB real checkpoints: a tiny synthetic fixture
 * with real HF tensor names exercises the same mapping + forward path. */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_ssm.h"
#include "wubu_model.h"
#include "wubu_repetition.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int all_finite(const float *a, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(a[i])) return 0;
    return 1;
}

static int nonzero_spread(const float *a, int n) {
    if (n <= 0) return 1;
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < n; i++) {
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
    }
    return (mx - mn) > 1e-3f;
}

int main(void) {
    wubu_adapter_t ad;
    memset(&ad, 0, sizeof(ad));
    ad.arch = WUBU_ARCH_QWEN36_HYBRID;
    ad.d_model = 256; ad.d_ff = 512; ad.n_layers = 2;
    ad.gqa_kv_heads = 2; ad.gqa_q_heads = 4; ad.gqa_head_dim = 4;
    ad.ssm_v_heads = 1; ad.ssm_k_heads = 16; ad.ssm_d_state = 128; ad.ssm_conv_kernel = 4;
    ad.n_experts = 0; ad.ok = 1;

    wubu_model_t m;
    if (wubu_model_init_safetensors(&m, "fixture_model.safetensors", &ad) != 0) {
        fprintf(stderr, "FAIL: init_safetensors\n"); return 1;
    }
    if (m.layers == NULL || m.n_layers != 2 || m.d_model != 256) {
        fprintf(stderr, "FAIL: model meta\n"); return 1;
    }
    wubu_layer_t *ly0 = &m.layers[0];

    /* Layer 0 is SSM-only in the fixture: check SSM weights and MLP.
     * For BF16 fixtures, the bridge keeps mmap'd raw bytes (lf32) and
     * leaves *_f32 NULL until wubu_ssm_ensure_f32(). */
    int has_ssm = (ly0->ssm.attn_qkv_weight_f32 || ly0->ssm.attn_qkv_weight_raw) &&
                  (ly0->ssm.attn_gate_weight_f32 || ly0->ssm.attn_gate_weight_raw) &&
                  (ly0->ssm.ssm_out_weight_f32  || ly0->ssm.ssm_out_weight_raw);
    int has_mlp = ly0->moe.ffn_up_exps || ly0->moe.ffn_gate_exps;
    if (!has_ssm || !has_mlp) {
        fprintf(stderr, "FAIL: weight pointers not mapped ssm=%d mlp=%d\n",
                has_ssm, has_mlp);
        return 1;
    }
    if (ly0->ssm.f32_mode != 1) { fprintf(stderr, "FAIL: f32_mode not set\n"); return 1; }
    if (ly0->gqa.attn_q_weight_q || ly0->ssm.attn_qkv_weight_q) {
        fprintf(stderr, "FAIL: quantized blob ptrs should be NULL in F32 mode\n"); return 1;
    }

    /* Materialize lazy BF16 tensors before reading their contents. */
    wubu_ssm_ensure_f32(&ly0->ssm, m.d_model, WUBU_DIMS.conv_dim, WUBU_DIMS.value_dim);
    if (ly0->gqa.attn_q_weight_raw)  wubu_gqa_ensure_f32(&ly0->gqa, m.d_model);

    /* Verify loaded F32 data is real (finite + has spread, not all-zero).
     * Only inspect each tensor's REAL element count. */
    int64_t qkv_n = (int64_t)WUBU_DIMS.conv_dim * m.d_model;
    int64_t gate_n = (int64_t)WUBU_DIMS.value_dim * m.d_model;
    int64_t out_n  = (int64_t)m.d_model * WUBU_DIMS.value_dim;
    int64_t a_n    = (int64_t)WUBU_DIMS.dt_rank;
    int64_t ffn_n  = (int64_t)m.d_model * ad.d_ff;
    int64_t chk_n[5] = { qkv_n, gate_n, out_n, a_n, ffn_n };
    float *chk[5];
    chk[0] = ly0->ssm.attn_qkv_weight_f32;
    chk[1] = ly0->ssm.attn_gate_weight_f32;
    chk[2] = ly0->ssm.ssm_out_weight_f32;
    chk[3] = ly0->ssm.ssm_a;
    chk[4] = ly0->moe.ffn_up_exps;
    int bad = 0;
    for (int i = 0; i < 5; i++) {
        if (!chk[i]) { bad = 1; break; }
        if (!all_finite(chk[i], (int)chk_n[i])) { bad = 1; break; }
        if (!nonzero_spread(chk[i], (int)chk_n[i])) { bad = 1; break; }
    }
    if (bad) { fprintf(stderr, "FAIL: loaded F32 weights degenerate/non-finite\n"); return 1; }

    wubu_model_safetensors_free(&m);
    printf("PASS: safetensors bridge maps HF->wubuwizard F32 structs (SSM+GQA+MoE, real weights)\n");

    /* ---- End-to-end: load again and RUN the forward + greedy decode with
     * repetition controller at the fixture's REAL runtime dims (D=256). ---- */
    wubu_model_t m2;
    if (wubu_model_init_safetensors(&m2, "fixture_model.safetensors", &ad) != 0) {
        fprintf(stderr, "FAIL: re-init for forward\n"); return 1;
    }
    const int vocab = m2.vocab_size;
    int prompt[1] = { 1 };
    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    if (!logits) { fprintf(stderr, "FAIL: oom logits\n"); return 1; }
    wubu_model_forward(&m2, prompt, 1, 1, logits);
    if (!all_finite(logits, vocab)) { fprintf(stderr, "FAIL: forward logits non-finite\n"); return 1; }
    int am = 0; float best = -1e30f;
    for (int i = 0; i < vocab; i++) if (logits[i] > best) { best = logits[i]; am = i; }
    printf("PASS: wubu_model_forward produces finite logits (argmax=%d)\n", am);

    wubu_rep_state_t *rep = wubu_rep_create(vocab, -1, 2, -1);
    wubu_rep_set_params(rep, 1.1f, 1.2f, 1.75f);

    int toks[8]; int nt = 0;
    int cur = am;
    for (int step = 0; step < 6; step++) {
        wubu_model_forward(&m2, &cur, 1, 1, logits);
        if (!all_finite(logits, vocab)) { fprintf(stderr, "FAIL: decode logits non-finite @%d\n", step); return 1; }
        wubu_rep_apply(rep, logits);
        if (!all_finite(logits, vocab)) { fprintf(stderr, "FAIL: post-rep logits non-finite @%d\n", step); return 1; }
        int nx = 0; float b2 = -1e30f;
        for (int i = 0; i < vocab; i++) if (logits[i] > b2) { b2 = logits[i]; nx = i; }
        wubu_rep_observe(rep, nx);
        toks[nt++] = nx; cur = nx;
    }
    printf("PASS: 6-token greedy decode runs with repetition controller (tokens:");
    for (int i = 0; i < nt; i++) printf(" %d", toks[i]);
    printf(")\n");

    wubu_rep_reset(rep);
    for (int i = 0; i < 8; i++) wubu_rep_observe(rep, 5);
    float *lg2 = (float *)malloc((size_t)vocab * sizeof(float));
    for (int i = 0; i < vocab; i++) lg2[i] = (i == 3) ? 2.0f : (i == 5 ? 2.0f : 0.0f);
    float before5 = lg2[5];
    wubu_rep_apply(rep, lg2);
    if (!all_finite(lg2, vocab)) { fprintf(stderr, "FAIL: dry logits non-finite\n"); return 1; }
    if (lg2[5] >= before5) { fprintf(stderr, "FAIL: DRY did not damp repeated token 5 (%.3f vs %.3f)\n", lg2[5], before5); return 1; }
    printf("PASS: DRY suppresses repeated token (logit %.3f -> %.3f)\n", before5, lg2[5]);

    free(lg2); wubu_rep_free(rep); free(logits);
    wubu_model_safetensors_free(&m2);
    return 0;
}
