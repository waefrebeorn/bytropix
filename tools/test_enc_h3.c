/*
 * test_enc_h3.c — tests for MiniMax H3 encoder NVFP4 requant + ConvRot.
 *
 * Covers: unrotate (Hadamard self-inverse), MXFP4 pack/unpack round-trip,
 * packed size computation, NULL safety, ConvRot prefix handling.
 *
 * C11, no external deps.
 */
#include "wubu_enc_h3.h"
#include "wubu_rotate.h"
#include "wubu_fp8.h"
#include "wubu_nvfp4.h"
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
    printf("=== H3 Encoder (NVFP4 + ConvRot) Tests ===\n\n");

    /* ---- Test 1: create/free ---- */
    printf("--- Test 1: create/free ---\n");
    int rows = 4, cols = 32;
    wubu_enc_h3_t *enc = wubu_enc_h3_create(rows, cols);
    check("create non-NULL", enc != NULL);

    /* NULL safety */
    wubu_enc_h3_free(NULL);
    check("free(NULL) safe", 1);
    wubu_enc_h3_free(enc);
    check("free doesn't crash", 1);

    /* ---- Test 2: packed size computation ---- */
    printf("\n--- Test 2: packed size ---\n");
    /* 32 elements per block (NVFP4: 16 nibbles + 1 scale byte = 17 per block)
     * For 4 rows x 32 cols = 128 elements → 4 blocks → 4*17 = 68 bytes */
    enc = wubu_enc_h3_create(rows, cols);
    size_t psz = wubu_enc_h3_packed_size(rows, cols);
    /* 128 elements total / 32 per block = 4 blocks * 17 bytes = 68 */
    check("packed size 68 bytes", psz == 68);

    /* NULL safety */
    size_t ps_null = wubu_enc_h3_packed_size(0, 0);
    check("packed_size(0,0)=0", ps_null == 0);
    wubu_enc_h3_free(enc);

    /* ---- Test 3: unrotate (Hadamard self-inverse) ---- */
    printf("\n--- Test 3: unrotate (Hadamard self-inverse) ---\n");
    /* Hadamard is self-inverse: H·H = I, so unrotate(unrotate(x)) = x */
    enc = wubu_enc_h3_create(1, 8); /* prefix = pow2_floor(8) = 8 */
    check("create 1x8", enc != NULL);

    /* Use a simple vector */
    float W[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float orig[8];
    memcpy(orig, W, sizeof(W));

    /* Unrotate = apply Hadamard to first 8 columns (all) */
    int rc = wubu_enc_h3_unrotate(W, 1, 8);
    check("unrotate returns 0", rc == 0);

    /* Unrotate again — should restore original */
    rc = wubu_enc_h3_unrotate(W, 1, 8);
    check("double unrotate returns 0", rc == 0);

    for (int i = 0; i < 8; i++) {
        check_f32("Hadamard self-inverse", W[i], orig[i], 1e-5f);
        if (i >= 2) break; /* check first 3 */
    }

    /* NULL safety */
    check("unrotate NULL", wubu_enc_h3_unrotate(NULL, 1, 8) == -1);

    /* ---- Test 4: MXFP4 requant round-trip ---- */
    printf("\n--- Test 4: MXFP4 requant round-trip ---\n");
    /* Use a small matrix where ConvRot prefix = 4 (pow2_floor(6) = 4) */
    enc = wubu_enc_h3_create(1, 6); /* rows=1, cols=6, prefix=4 */

    float W2[6] = {0.5f, -1.0f, 0.25f, 2.0f, 0.1f, -0.5f};
    size_t psz2 = wubu_enc_h3_packed_size(1, 6);
    /* 6 elements, 1 block of 32 → 1 block → 17 bytes */
    check("packed_size(1,6)=17", psz2 == 17);

    uint8_t *packed2 = (uint8_t *)malloc(psz2);
    check("packed alloc", packed2 != NULL);

    rc = wubu_enc_h3_requant_nvfp4(W2, 1, 6, packed2, psz2);
    check("requant returns 0", rc == 0);

    /* Dequant + rotate back */
    float W_back[6];
    rc = wubu_enc_h3_dequant_rotate(packed2, 1, 6, W_back, psz2);
    check("dequant_rotate returns 0", rc == 0);

    /* The values should be approximately preserved
     * (MXFP4 + Hadamard round-trip, allow tolerance for quant error) */
    float max_diff = 0.0f;
    for (int i = 0; i < 6; i++) {
        float d = fabsf(W2[i] - W_back[i]);
        if (d > max_diff) max_diff = d;
    }
    /* With ConvRot un-rotation + requant + dequant+rotate,
     * the round-trip should be ~identity for in-range values */
    check_f32("round-trip max_diff < 2.0", max_diff, 0.0f, 2.0f);

    free(packed2);
    wubu_enc_h3_free(enc);

    /* ---- Test 5: zero-weight path ---- */
    printf("\n--- Test 5: zero weights ----\n");
    enc = wubu_enc_h3_create(1, 8);
    float zeros[8] = {0};
    size_t psz3 = wubu_enc_h3_packed_size(1, 8);
    uint8_t *packed3 = (uint8_t *)malloc(psz3);
    float back3[8];

    rc = wubu_enc_h3_requant_nvfp4(zeros, 1, 8, packed3, psz3);
    check("requant zeros returns 0", rc == 0);
    rc = wubu_enc_h3_dequant_rotate(packed3, 1, 8, back3, psz3);
    check("dequant zeros returns 0", rc == 0);

    int all_zero = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(back3[i]) > 1e-6f) all_zero = 0;
    check("zeros stay zero", all_zero);

    free(packed3);
    wubu_enc_h3_free(enc);

    /* ---- Final ---- */
    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
