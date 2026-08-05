/*
 * test_megakernel.c — tests for Photon 2.0 fused decode megakernel.
 *
 * Covers: create/free, fused decode single token, NULL safety,
 * config validation.
 *
 * C11, no external deps.
 */
#include "wubu_megakernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

int main(void) {
    printf("=== Photon 2.0 Megakernel Tests ===\n\n");

    /* ---- Test 1: create/free ---- */
    printf("--- Test 1: create/free ---\n");
    wubu_megakernel_cfg_t cfg = {
        .d_model = 8, .n_heads = 4, .n_kv_heads = 2,
        .d_head = 2, .d_ff = 16, .rms_eps = 1, .rms_epsilon = 1e-6f
    };
    wubu_megakernel_t *mk = wubu_megakernel_create(&cfg);
    check("create non-NULL", mk != NULL);

    /* Invalid: n_heads must be divisible by n_kv_heads */
    wubu_megakernel_cfg_t bad = { .d_model=8, .n_heads=5, .n_kv_heads=2, .d_head=2, .d_ff=16, .rms_eps=1, .rms_epsilon=1e-6f };
    check("reject n_heads %% n_kv_heads != 0", wubu_megakernel_create(&bad) == NULL);

    /* Invalid: d_model <= 0 */
    bad = (wubu_megakernel_cfg_t){ .d_model=0, .n_heads=4, .n_kv_heads=2, .d_head=2, .d_ff=16, .rms_eps=1, .rms_epsilon=1e-6f };
    check("reject d_model=0", wubu_megakernel_create(&bad) == NULL);

    /* Invalid: NULL cfg */
    check("reject NULL cfg", wubu_megakernel_create(NULL) == NULL);

    wubu_megakernel_free(mk);
    check("free doesn't crash", 1);
    wubu_megakernel_free(NULL);
    check("free(NULL) safe", 1);

    /* ---- Test 2: fused decode (identity weights) ---- */
    printf("\n--- Test 2: fused decode (identity) ---\n");
    mk = wubu_megakernel_create(&cfg);
    check("create for decode", mk != NULL);

    int D = 8, DHF = 16, NKV = 2, DH = 2;
    /* QKV: [Q: D*D | K: D*(NKV*DH) | V: D*(NKV*DH)] = [D*D + 2*D*NKV*DH] */
    int qkv_size = D * D + 2 * D * NKV * DH;  /* 64 + 2*8*2*2 = 64+32 = 96 */
    float *ctx = (float *)calloc(D, sizeof(float));
    float *qkv_w = (float *)calloc(qkv_size, sizeof(float));
    float *attn_w = (float *)calloc(D * D, sizeof(float));
    float *ffh_w = (float *)calloc(DHF * D, sizeof(float));
    float *ffo_w = (float *)calloc(D * DHF, sizeof(float));
    float *rms1 = (float *)calloc(D, sizeof(float));
    float *rms2 = (float *)calloc(D, sizeof(float));
    /* KV cache: [2 * NKV * DH * 1] — K cache then V cache (each NKV*DH*1) */
    float *kv = (float *)calloc(2 * NKV * DH * 1, sizeof(float));
    float *out = (float *)calloc(D, sizeof(float));

    check("alloc", ctx && qkv_w && attn_w && ffh_w && ffo_w && rms1 && rms2 && kv && out);

    /* Set RMSNorm to 1.0 (identity) */
    for (int i = 0; i < D; i++) { rms1[i] = 1.0f; rms2[i] = 1.0f; }

    /* Set QKV to identity for Q part (first D*D elements) */
    for (int i = 0; i < D; i++) qkv_w[i * D + i] = 1.0f;
    /* K and V parts: set to identity-ish (so K[i,j] = delta[i,j]) */
    for (int i = 0; i < NKV * DH; i++) qkv_w[D * D + i * D + i] = 1.0f;

    /* Set attn output to identity */
    for (int i = 0; i < D; i++) attn_w[i * D + i] = 1.0f;

    /* Set FFN: ffh = identity (DHF x D, select first D rows), ffo = identity (D x DHF) */
    for (int i = 0; i < D; i++) ffh_w[i * D + i] = 1.0f;
    for (int i = 0; i < D; i++) ffo_w[i * DHF + i] = 1.0f;

    /* Input: non-zero so RMSNorm output is non-trivial */
    for (int i = 0; i < D; i++) ctx[i] = (float)(i + 1) * 0.1f;

    int rc = wubu_megakernel_decode(mk, ctx, qkv_w, attn_w, ffh_w, ffo_w,
                                     rms1, rms2, kv, 0, out);
    check("decode returns 0", rc == 0);

    /* NULL safety */
    check("decode NULL ctx", wubu_megakernel_decode(mk, NULL, qkv_w, attn_w, ffh_w, ffo_w, rms1, rms2, kv, 0, out) == -1);

    /* Output should be finite (not NaN/Inf) */
    int all_finite = 1;
    for (int i = 0; i < D; i++) {
        if (!isfinite(out[i])) { all_finite = 0; break; }
    }
    check("output all finite", all_finite);

    free(ctx); free(qkv_w); free(attn_w); free(ffh_w); free(ffo_w);
    free(rms1); free(rms2); free(kv); free(out);
    wubu_megakernel_free(mk);

    /* ---- Final ---- */
    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
