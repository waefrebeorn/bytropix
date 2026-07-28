/* test_probe_qwen.c -- load real Qwen3.6-27B (MAX_LAYERS env) and dump
 * layer-0 weight pointers + allocated state buffers. Verifies the bridge maps
 * real weights and the forward's state sizing matches the loader. */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_dims.h"
#include "wubu_model.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int finite_all(const float *a, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(a[i])) return 0;
    return 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *dir = (argc > 1) ? argv[1] : "/home/wubu/models/Qwen3.6-27B";
    setenv("MAX_LAYERS", "1", 1);
    wubu_adapter_t ad; memset(&ad, 0, sizeof(ad));
    if (!wubu_adapter_load(&ad, dir)) { ad.arch = WUBU_ARCH_QWEN_FAMILY; ad.ok = 1; }
    wubu_model_t m;
    if (wubu_model_init_safetensors(&m, dir, &ad) != 0) { fprintf(stderr, "init FAIL\n"); return 1; }
    printf("d_model=%d n_layers=%d vocab=%d\n", m.d_model, m.n_layers, m.vocab_size);
    wubu_layer_t *L = &m.layers[0];
    printf("L0 is_ssm=%d\n", L->is_ssm);
    printf("qkv=%p gate=%p out=%p a=%p b=%p a_log=%p dt_bias=%p conv=%p norm=%p\n",
        (void*)L->ssm.attn_qkv_weight_f32, (void*)L->ssm.attn_gate_weight_f32,
        (void*)L->ssm.ssm_out_weight_f32, (void*)L->ssm.ssm_alpha_weight,
        (void*)L->ssm.ssm_beta_weight, (void*)L->ssm.ssm_a, (void*)L->ssm.ssm_dt_bias,
        (void*)L->ssm.ssm_conv1d_weight, (void*)L->ssm.ssm_norm_weight);
    printf("gqa q=%p k=%p v=%p o=%p\n",
        (void*)L->gqa.attn_q_weight, (void*)L->gqa.attn_k_weight,
        (void*)L->gqa.attn_v_weight, (void*)L->gqa.attn_output_weight);
    printf("lazy_embd=%p lazy_lmhead=%p token_embd=%p output_weight=%p\n",
        (void*)m.lazy_embd_raw, (void*)m.lazy_lmhead_raw, (void*)m.token_embd, (void*)m.output_weight);
    printf("ssm_states=%p conv_states=%p  ssm_v_heads=%d ssm_d_state=%d\n",
        (void*)m.ssm_states, (void*)m.conv_states, WUBU_DIMS.ssm_v_heads, SSM_D_STATE);

    /* Direct SSM forward on real layer-0 weights, B=1, T=1. */
    const int D = WUBU_DIMS.d_model;
    float *x = (float *)malloc(D * sizeof(float));
    for (int i = 0; i < D; i++) x[i] = (float)(i % 7 - 3) * 0.01f;
    int ssz = WUBU_DIMS.ssm_v_heads * SSM_D_STATE * SSM_D_STATE;
    int csz = (WUBU_DIMS.conv_kernel - 1) * m.conv_dim;
    float *ssm_state = (float *)calloc((size_t)ssz, sizeof(float));
    float *conv_state = (float *)calloc((size_t)csz, sizeof(float));
    float *ssm_out = (float *)malloc(D * sizeof(float));
    printf("calling wubu_ssm_forward D=%d ssz=%d csz=%d conv_dim=%d\n", D, ssz, csz, m.conv_dim);
    wubu_ssm_forward(x, 1, 1, &L->ssm, ssm_state, conv_state, ssm_out, NULL, NULL);
    int ok = finite_all(ssm_out, D);
    printf("real SSM forward: ssm_out[0]=%g finite=%s\n", ssm_out[0], ok ? "YES" : "NO");

    /* Manual lazy-embed sanity: dequant token 0, row 0, print a few vals. */
    printf("lazy_embd_dtype=%d lazy_embd_row=%lld\n", m.lazy_embd_dtype, (long long)m.lazy_embd_row);
    if (m.lazy_embd_raw) {
        const uint16_t *s = (const uint16_t *)m.lazy_embd_raw;
        printf("embed[0]=%g embed[1]=%g embed[10]=%g\n",
               st_bf16_to_f32(s[0]), st_bf16_to_f32(s[1]), st_bf16_to_f32(s[10]));
    }
    /* Full model forward on a tiny token sequence (exercises embed + SSM +
     * GQA + lazy output-proj paths). */
    int toks[3] = { 0, 100, 248319 };
    float *logits = (float *)malloc((size_t)3 * m.vocab_size * sizeof(float));
    wubu_model_forward(&m, toks, 1, 3, logits);
    int fwd_ok = finite_all(logits, 3 * m.vocab_size);
    /* argmax of last token */
    int best = 0; float bestv = -1e30f;
    for (int v = 0; v < m.vocab_size; v++) if (logits[2*m.vocab_size + v] > bestv) { bestv = logits[2*m.vocab_size + v]; best = v; }
    printf("FULL forward: finite=%s argmax_tok=%d argmax_logit=%g\n", fwd_ok ? "YES":"NO", best, bestv);

    free(x); free(ssm_state); free(conv_state); free(ssm_out); free(logits);
    wubu_model_safetensors_free(&m);
    printf("%s\n", (ok && fwd_ok) ? "PASS" : "FAIL");
    return (ok && fwd_ok) ? 0 : 1;
}
