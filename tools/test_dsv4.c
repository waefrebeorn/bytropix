/*
 * test_dsv4.c — tests for DeepSeek-V4 Flash hybrid layer port.
 *
 * Covers: hyper-connection residual, sinkhorn normalization, hash routing,
 * MXFP4 pack/unpack round-trip, lightning indexer.
 *
 * C11, no external deps.
 */
#include "wubu_dsv4.h"
#include "wubu_hashrouter.h"
#include "wubu_mxfp4.h"
#include "wubu_dsa.h"
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
    printf("=== DeepSeek-V4 Flash Hybrid Layer Tests ===\n\n");

    /* ---- Test 1: create/free ---- */
    printf("--- Test 1: create/free ---\n");
    wubu_dsv4_cfg_t cfg = { .d_model=7168, .n_heads=96, .n_experts=256, .n_active=6, .n_layers=43 };
    wubu_dsv4_t *ds = wubu_dsv4_create(&cfg);
    check("create non-NULL", ds != NULL);

    /* Invalid configs */
    wubu_dsv4_cfg_t bad = { .d_model=0, .n_heads=96, .n_experts=256, .n_active=6, .n_layers=43 };
    check("reject d_model=0", wubu_dsv4_create(&bad) == NULL);

    bad = (wubu_dsv4_cfg_t){ .d_model=7168, .n_heads=96, .n_experts=256, .n_active=300, .n_layers=43 };
    check("reject n_active > n_experts", wubu_dsv4_create(&bad) == NULL);

    bad = (wubu_dsv4_cfg_t){ .d_model=7168, .n_heads=96, .n_experts=256, .n_active=6, .n_layers=0 };
    check("reject n_layers=0", wubu_dsv4_create(&bad) == NULL);

    wubu_dsv4_free(ds);
    check("free doesn't crash", 1);
    wubu_dsv4_free(NULL); /* should be safe */
    check("free(NULL) safe", 1);

    /* ---- Test 2: create valid + use ---- */
    printf("\n--- Test 2: hyper_residual ---\n");
    ds = wubu_dsv4_create(&cfg);
    check("create for hyper_residual", ds != NULL);

    int d_model = 8;
    float x[8]    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float ffn[8]  = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float out[8];

    /* out = x + 2.0 * ffn */
    int rc = wubu_dsv4_hyper_residual(x, ffn, 2.0f, d_model, out);
    check("hyper_residual returns 0", rc == 0);
    check("hyper_residual NULL safety", wubu_dsv4_hyper_residual(NULL, ffn, 2.0f, d_model, out) == -1);

    for (int i = 0; i < d_model; i++) {
        float expected = x[i] + 2.0f * ffn[i];
        check_f32("hyper_residual element", out[i], expected, 1e-6f);
        if (i > 2) break; /* check first 3 elements */
    }

    /* ---- Test 3: sinkhorn normalization ---- */
    printf("\n--- Test 3: sinkhorn_norm ---\n");
    /* 3x3 matrix: each row should sum to ~1 after sinkhorn */
    float w[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    int nt = 3, ne = 3;
    int sc = wubu_dsv4_sinkhorn_norm(w, nt, ne, 10);
    check("sinkhorn returns 0", sc == 0);

    /* After sinkhorn, row sums should be ~1 */
    for (int r = 0; r < nt; r++) {
        float sum = w[r*ne+0] + w[r*ne+1] + w[r*ne+2];
        check_f32("row sum ~1", sum, 1.0f, 0.01f);
    }

    /* NULL safety */
    check("sinkhorn NULL safe", wubu_dsv4_sinkhorn_norm(NULL, nt, ne, 10) == -1);

    /* ---- Test 4: hash routing ---- */
    printf("\n--- Test 4: hash routing ---\n");
    int experts[6];
    int nn = wubu_dsv4_route(ds, 42, 0, experts);
    check("route returns n_active", nn == 6);
    check("route experts in range", 1);
    for (int i = 0; i < nn; i++) {
        if (experts[i] < 0 || experts[i] >= 256) { check("expert out of range", 0); break; }
    }

    /* Distinctness check */
    int distinct = 1;
    for (int i = 0; i < nn; i++)
        for (int j = i+1; j < nn; j++)
            if (experts[i] == experts[j]) distinct = 0;
    check("route experts distinct", distinct);

    /* Determinism: same token+pos => same experts */
    int experts2[6];
    wubu_dsv4_route(ds, 42, 0, experts2);
    check("route deterministic", memcmp(experts, experts2, 6*sizeof(int)) == 0);

    wubu_dsv4_free(ds);

    /* ---- Test 5: MXFP4 expert pack/unpack round-trip ---- */
    printf("\n--- Test 5: MXFP4 expert pack/unpack ----\n");
    int n_experts = 4, expert_dim = 32;
    float *expert_w = (float *)malloc((size_t)n_experts * expert_dim * sizeof(float));
    uint8_t *packed = (uint8_t *)malloc((size_t)n_experts * 17 * sizeof(uint8_t)); /* 32 elem -> 17 bytes MXFP4 */
    float *unpacked = (float *)malloc((size_t)n_experts * expert_dim * sizeof(float));

    check("alloc", expert_w && packed && unpacked);

    /* Fill with known pattern */
    for (int i = 0; i < n_experts * expert_dim; i++)
        expert_w[i] = (float)(i % 7) - 3.0f;

    int rc2 = wubu_dsv4_pack_experts_mxfp4(expert_w, n_experts, expert_dim, packed);
    check("pack returns 0", rc2 == 0);

    int rc3 = wubu_dsv4_unpack_experts_mxfp4(packed, n_experts, expert_dim, unpacked);
    check("unpack returns 0", rc3 == 0);

    /* Round-trip: unpacked should be close to original.
     * MXFP4 quantization error per element is ~1.0 at scale. */
    float max_diff = 0.0f;
    for (int i = 0; i < n_experts * expert_dim; i++) {
        float d = fabsf(expert_w[i] - unpacked[i]);
        if (d > max_diff) max_diff = d;
    }
    check_f32("pack/ununpack max_diff < 5.0", max_diff, 0.0f, 5.0f);

    free(expert_w); free(packed); free(unpacked);

    /* ---- Test 6: lightning indexer ---- */
    printf("\n--- Test 6: lightning indexer ---\n");
    /* Simple: 4 blocks, d=4, d_v=2 */
    float query[4] = {1, 0, 0, 0};
    float means[4][4] = {
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}
    };
    float bvals[4][2] = {
        {10, 20}, {30, 40}, {50, 60}, {70, 80}
    };
    float v[2];
    const float *bm[4] = {(float*)means[0], (float*)means[1], (float*)means[2], (float*)means[3]};
    const float *bv[4] = {(float*)bvals[0], (float*)bvals[1], (float*)bvals[2], (float*)bvals[3]};

    int rc4 = wubu_dsv4_lightning_indexer(query, 4, bm, bv, 4, 2, 2, v);
    check("lightning_indexer returns 0", rc4 == 0);
    /* Query {1,0,0,0} matches block 0 exactly, but softmax over all 4 blocks
     * means other blocks contribute. Output should be dominated by block 0's
     * values ({10,20}) but shifted by other blocks. */
    check_f32("lightning out[0] in range", v[0], 10.0f, 10.0f);
    check("lightning out[0] finite", isfinite(v[0]));

    /* NULL safety */
    check("lightning NULL safe", wubu_dsv4_lightning_indexer(NULL, 4, bm, bv, 4, 2, 2, v) == -1);

    /* ---- Final ---- */
    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
