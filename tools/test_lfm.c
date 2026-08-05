/*
 * test_lfm.c — tests for LFM2.5-2.6B hybrid attention layer.
 *
 * Covers: create/free, linear attention (DeltaNet), softmax GQA,
 * hybrid layer composition, NULL safety.
 *
 * C11, no external deps.
 */
#include "wubu_lfm.h"
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

static void check_f32(const char *name, float got, float expected, float tol) {
    tests_run++;
    float diff = fabsf(got - expected);
    if (diff <= tol) { tests_pass++; printf("  PASS: %s (got %.6f, exp %.6f)\n", name, got, expected); }
    else             { printf("  FAIL: %s (got %.6f, exp %.6f, diff %.2e)\n", name, got, expected, diff); }
}

int main(void) {
    printf("=== LFM2.5 Hybrid Attention Tests ===\n\n");

    /* Use d_model=4, n_heads=2, d_head=2, n_kv_heads=1 so all test arrays match. */
    int d_model = 4;
    int n_heads = 2, d_head = 2, n_kv_heads = 1;

    /* ---- Test 1: create/free ---- */
    printf("--- Test 1: create/free ---\n");
    wubu_lfm_cfg_t cfg = { .d_model=d_model, .n_heads=n_heads, .d_head=d_head,
                           .n_kv_heads=n_kv_heads, .n_layers=30, .hybrid_gate=1 };
    wubu_lfm_t *lfm = wubu_lfm_create(&cfg);
    check("create non-NULL", lfm != NULL);

    /* Invalid: NULL cfg */
    check("reject NULL cfg", wubu_lfm_create(NULL) == NULL);

    /* Invalid: n_heads % n_kv_heads != 0 */
    wubu_lfm_cfg_t bad = { .d_model=128, .n_heads=7, .d_head=16, .n_kv_heads=4, .n_layers=30, .hybrid_gate=1 };
    check("reject n_heads %% n_kv_heads != 0", wubu_lfm_create(&bad) == NULL);

    wubu_lfm_free(lfm);
    check("free doesn't crash", 1);
    wubu_lfm_free(NULL);
    check("free(NULL) safe", 1);

    /* ---- Test 2: linear attention (DeltaNet) ---- */
    printf("\n--- Test 2: linear attention (DeltaNet) ---\n");
    lfm = wubu_lfm_create(&cfg);
    check("create for linear_attn", lfm != NULL);

    /* Identity state S (d x d) => out = S @ k = k */
    float S[16]; /* 4x4 identity */
    memset(S, 0, sizeof(S));
    S[0] = S[5] = S[10] = S[15] = 1.0f;

    float k_lin[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_lin[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float Sout[16], lin_out[4];
    float beta = 0.5f;

    int rc = wubu_lfm_linear_attn(S, k_lin, v_lin, d_model, beta, Sout, lin_out);
    check("linear_attn returns 0", rc == 0);

    /* DeltaNet: S' = S - beta*(S·k - v) * k^T, then out = S' · k
     * With S=I, k={1,2,3,4}, v={10,20,30,40}, beta=0.5:
     *   S·k = {1,2,3,4}, delta = {1-10, 2-20, 3-30, 4-40} = {-9,-18,-27,-36}
     *   k^T·k = 1+4+9+16 = 30
     *   out = k - 0.5 * delta * 30 = {1+135, 2+270, 3+405, 4+540} = {136,272,408,544} */
    check_f32("linear_out[0]=136", lin_out[0], 136.0f, 1e-3f);
    check_f32("linear_out[1]=272", lin_out[1], 272.0f, 1e-3f);
    check_f32("linear_out[2]=408", lin_out[2], 408.0f, 1e-3f);
    check_f32("linear_out[3]=544", lin_out[3], 544.0f, 1e-3f);

    /* NULL safety */
    check("linear_attn NULL ctx", wubu_lfm_linear_attn(NULL, k_lin, v_lin, d_model, beta, Sout, lin_out) == -1);
    check("linear_attn NULL k", wubu_lfm_linear_attn(S, NULL, v_lin, d_model, beta, Sout, lin_out) == -1);

    /* ---- Test 3: softmax GQA attention ---- */
    printf("\n--- Test 3: softmax GQA attention ---\n");
    /* queries: [n_heads * d_head] = [4], keys/values: [n_kv_heads * d_head * seq_len] = [2] */
    float q[4] = {1.0f, 1.0f, 2.0f, 2.0f};
    float k_arr[2] = {1.0f, 0.0f};
    float v_arr[2] = {5.0f, 6.0f};
    float sm_out[4];

    rc = wubu_lfm_softmax_attn(q, k_arr, v_arr, n_heads, n_kv_heads, d_head, 1, 0, sm_out);
    check("softmax_attn returns 0", rc == 0);

    /* With seq_len=1: softmax = 1.0, attn_out = V broadcast to all heads */
    check_f32("softmax out[0]=5", sm_out[0], 5.0f, 1e-6f);
    check_f32("softmax out[1]=6", sm_out[1], 6.0f, 1e-6f);
    check_f32("softmax out[2]=5", sm_out[2], 5.0f, 1e-6f);
    check_f32("softmax out[3]=6", sm_out[3], 6.0f, 1e-6f);

    /* NULL safety */
    check("softmax NULL q", wubu_lfm_softmax_attn(NULL, k_arr, v_arr, n_heads, n_kv_heads, d_head, 1, 0, sm_out) == -1);

    /* ---- Test 4: hybrid layer ---- */
    printf("\n--- Test 4: hybrid layer ---\n");
    /* Identity linear_state => linear path: DeltaNet update
     * With S=I, k_lin={0.1,0.2,0.3,0.4}, v_lin={1,2,3,4}, beta=0.9:
     *   S·k = k = {0.1,0.2,0.3,0.4}, delta = k - v = {-0.9,-1.8,-2.7,-3.6}
     *   k^T·k = 0.01+0.04+0.09+0.16 = 0.30
     *   linear_out = k - 0.9 * delta * 0.30 = {0.1+0.243, 0.2+0.486, 0.3+0.729, 0.4+0.972}
     *             = {0.343, 0.686, 1.029, 1.372}
     * Softmax path: out = {5,6,5,6} (V broadcast, seq_len=1)
     * hybrid (gate=0.5, even layer): 0.5*linear + 0.5*softmax = {2.6465, 3.343, 2.7645, 3.686}
     */
    float ident_S[16];
    memset(ident_S, 0, sizeof(ident_S));
    ident_S[0] = ident_S[5] = ident_S[10] = ident_S[15] = 1.0f;
    float query_h[4] = {1, 1, 2, 2};    /* [n_heads * d_head] */
    float key_h[2]   = {1, 0};          /* [n_kv_heads * d_head * seq_len=1] */
    float val_h[2]   = {5, 6};
    float h_lin_k[4]  = {0.1f, 0.2f, 0.3f, 0.4f};
    float h_lin_v[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_Sout[16];
    float h_out[4];
    float gate = 0.5f;

    rc = wubu_lfm_hybrid_layer(lfm, ident_S, query_h, key_h, val_h,
                                h_lin_k, h_lin_v, gate, 0, h_out, h_Sout);
    check("hybrid returns 0", rc == 0);

    /* Even layer (layer_idx=0): gate=0.5 => out = 0.5*linear + 0.5*softmax */
    check_f32("hybrid out[0]~2.65", h_out[0], 2.6465f, 0.5f);
    check("hybrid out finite", isfinite(h_out[0]) && isfinite(h_out[1]) &&
                            isfinite(h_out[2]) && isfinite(h_out[3]));

    /* NULL safety */
    check("hybrid NULL out", wubu_lfm_hybrid_layer(lfm, ident_S, query_h, key_h, val_h, h_lin_k, h_lin_v, gate, 0, NULL, h_Sout) == -1);

    wubu_lfm_free(lfm);

    /* ---- Final ---- */
    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
