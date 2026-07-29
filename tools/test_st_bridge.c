/* test_st_bridge.c -- verify the safetensors->wubuwizard F32 bridge
 * maps the REAL published HF tensor names (model.language_model.layers.N...)
 * onto wubuwizard's SSM/GQA/MoE F32 weight structs with no gaps, AND runs the
 * full pipeline (load -> wubu_model_forward -> greedy decode -> repetition
 * controller) at the model's own runtime dims (D=16 fixture). The forward
 * reads D_MODEL from the runtime wubu_dims global, so it runs on any model.
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
    wubu_layer_t *ly = &m.layers[0];
    if (!ly->ssm.attn_qkv_weight_f32 || !ly->ssm.attn_gate_weight_f32 ||
        !ly->ssm.ssm_out_weight_f32 || !ly->ssm.ssm_a ||
        !ly->ssm.ssm_beta_weight || !ly->ssm.ssm_dt_bias ||
        !ly->ssm.ssm_conv1d_weight || !ly->ssm.ssm_norm_weight ||
        !ly->gqa.attn_q_weight || !ly->gqa.attn_k_weight ||
        !ly->gqa.attn_v_weight || !ly->gqa.attn_output_weight ||
        !ly->moe.ffn_up_exps) {
        fprintf(stderr, "FAIL: weight pointers not mapped qkv=%p gate=%p out=%p ssm_a=%p beta=%p dt=%p conv=%p norm=%p gqa_q=%p gqa_k=%p gqa_v=%p gqa_o=%p ffn_up=%p\n",
            (void*)ly->ssm.attn_qkv_weight_f32, (void*)ly->ssm.attn_gate_weight_f32,
            (void*)ly->ssm.ssm_out_weight_f32, (void*)ly->ssm.ssm_a,
            (void*)ly->ssm.ssm_beta_weight, (void*)ly->ssm.ssm_dt_bias,
            (void*)ly->ssm.ssm_conv1d_weight, (void*)ly->ssm.ssm_norm_weight,
            (void*)ly->gqa.attn_q_weight, (void*)ly->gqa.attn_k_weight,
            (void*)ly->gqa.attn_v_weight, (void*)ly->gqa.attn_output_weight,
            (void*)ly->moe.ffn_up_exps);
        return 1;
    }
    if (ly->ssm.f32_mode != 1) { fprintf(stderr, "FAIL: f32_mode not set\n"); return 1; }
    if (ly->gqa.attn_q_weight_q || ly->ssm.attn_qkv_weight_q) {
        fprintf(stderr, "FAIL: quantized blob ptrs should be NULL in F32 mode\n"); return 1;
    }

    /* Verify loaded F32 data is real (finite + has spread, not all-zero).
     * IMPORTANT: only inspect each tensor's REAL element count -- several
     * SSM tensors are tiny (DT_RANK=32 elements) and reading past them would
     * walk into adjacent heap (garbage / segfault), not the model weights. */
    int chk_n[7] = { 256*4192 /*qkv D*CONVD*/, 256*128 /*gate VD*D*/, 128*256 /*out VD*D*/,
                     32 /*ssm_a DT*/, 16*256 /*attn_q qh*hd*D*/, 256*256 /*attn_out D*D*/,
                     256*512 /*ffn_up D*dff*/ };
    int bad = 0;
    float *chk[] = { ly->ssm.attn_qkv_weight_f32, ly->ssm.attn_gate_weight_f32,
                    ly->ssm.ssm_out_weight_f32, ly->ssm.ssm_a,
                    ly->gqa.attn_q_weight, ly->gqa.attn_output_weight,
                    ly->moe.ffn_up_exps };
    for (int i = 0; i < 7; i++) {
        int n = chk_n[i];
        if (!all_finite(chk[i], n)) { bad = 1; break; }
        float mn = 1e30f, mx = -1e30f;
        for (int j = 0; j < n; j++) { if (chk[i][j] < mn) mn = chk[i][j]; if (chk[i][j] > mx) mx = chk[i][j]; }
        if (mx - mn < 1e-3f) { bad = 1; break; } /* degenerate => not real data */
    }
    if (bad) { fprintf(stderr, "FAIL: loaded F32 weights degenerate/non-finite\n"); return 1; }

    wubu_model_safetensors_free(&m);
    printf("PASS: safetensors bridge maps HF->wubuwizard F32 structs (SSM+GQA+MoE, real weights)\n");

    /* ---- End-to-end: load again and actually RUN the forward + a short
     * greedy decode loop with the repetition controller (your llama.cpp
     * DRY tuning). Proves load->forward->logits->generate is real, not a
     * form-without-function mapping. Runs at the fixture's REAL SSM dims
     * (D=256, VALUE_DIM=128, KEY_DIM=2048) via the runtime wubu_dims global. ---- */
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
    /* argmax over vocab */
    int am = 0; float best = -1e30f;
    for (int i = 0; i < vocab; i++) if (logits[i] > best) { best = logits[i]; am = i; }
    printf("PASS: wubu_model_forward produces finite logits (argmax=%d)\n", am);

    /* Repetition controller with the Agents-A1/Qwen3.6 DRY tuning:
     *   repeat_penalty 1.1, dry_mult 1.2, dry_base 1.75, ngram 2, whole ctx */
    wubu_rep_state_t *rep = wubu_rep_create(vocab, -1, 2, -1);
    wubu_rep_set_params(rep, 1.1f, 1.2f, 1.75f);

    /* Greedy decode 6 tokens; verify finite + distinct-ish output and that
     * repetition suppression does not NaN the logits. */
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

    /* Verify DRY actually damps a forced repeat: observe token 5 many times,
     * then confirm its logit is suppressed relative to an unseen token. */
    wubu_rep_reset(rep);
    /* build a context of repeated token 5 */
    for (int i = 0; i < 8; i++) wubu_rep_observe(rep, 5);
    float *lg2 = (float *)malloc((size_t)vocab * sizeof(float));
    for (int i = 0; i < vocab; i++) lg2[i] = (i == 3) ? 2.0f : (i == 5 ? 2.0f : 0.0f);
    float before5 = lg2[5];
    wubu_rep_apply(rep, lg2);
    if (!all_finite(lg2, vocab)) { fprintf(stderr, "FAIL: dry logits non-finite\n"); return 1; }
    if (lg2[5] >= before5) { fprintf(stderr, "FAIL: DRY did not damp repeated token 5 (%.3f vs %.3f)\n", lg2[5], before5); return 1; }
    printf("PASS: DRY suppresses repeated token (logit %.3f -> %.3f)\n", before5, lg2[5]);

    free(lg2);
    wubu_rep_free(rep);
    free(logits);
    wubu_model_safetensors_free(&m2);
    return 0;
}
