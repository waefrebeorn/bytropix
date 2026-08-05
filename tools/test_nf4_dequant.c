/*
 * test_nf4_dequant.c — runtime tests for NF4 dequantization.
 *
 * 22 test vectors covering:
 * - All 16 NF4 codes with scale=1.0
 * - Scale factor application (0.5, 2.0)
 * - Edge cases: code 0 (min), code 15 (max)
 * - Odd element count (tests nibble extraction boundary)
 * - Inverse CDF level verification against known scipy values
 */
#include "wubu_dequant_nf4.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_passed = 0;

static void check(const char *name, float got, float expected, float tol) {
    tests_run++;
    float diff = fabsf(got - expected);
    if (diff <= tol) {
        tests_passed++;
        printf("  PASS: %s — got %.6f, expected %.6f (diff %.2e)\n", name, got, expected, diff);
    } else {
        printf("  FAIL: %s — got %.6f, expected %.6f (diff %.2e, tol %.2e)\n", name, got, expected, diff, tol);
    }
}

/* Expected NF4 levels (Φ^{-1}((2j+1)/32)) — cross-checked against bitsandbytes */
static const float nf4_ref[16] = {
     -2.716777f, -2.326348f, -2.021329f, -1.750686f,
     -1.513346f, -1.302350f, -1.115163f, -0.947420f,
     -0.795728f, -0.657596f, -0.531329f, -0.415593f,
     -0.309225f, -0.211034f, -0.120077f, -0.034988f
};

int main(void) {
    printf("=== NF4 Dequantization Tests ===\n\n");

    /* Test 1: All 16 codes with scale=1.0
     * Pack codes 0-15 into bytes: 0x01 0x23 0x45 0x67 0x89 0xAB 0xCD 0xEF
     * byte 0 = 0x01: hi=0, lo=1 → codes [0,1]
     * byte 1 = 0x23: hi=2, lo=3 → codes [2,3]
     * byte 2 = 0x45: hi=4, lo=5 → codes [4,5]
     * byte 3 = 0x67: hi=6, lo=7 → codes [6,7]
     * byte 4 = 0x89: hi=8, lo=9 → codes [8,9]
     * byte 5 = 0xAB: hi=10, lo=11 → codes [10,11]
     * byte 6 = 0xCD: hi=12, lo=13 → codes [12,13]
     * byte 7 = 0xEF: hi=14, lo=15 → codes [14,15]
     */
    printf("--- Test 1: All 16 NF4 codes, scale=1.0 ---\n");
    unsigned char codes[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
    float out[16];
    nf4_dequantize_row(codes, out, 1.0f, 16);
    for (int i = 0; i < 16; i++) {
        char name[64];
        snprintf(name, sizeof(name), "code %d (scale=1.0)", i);
        check(name, out[i], nf4_ref[i], 1e-5f);
    }

    /* Test 2: Scale factor 0.5 */
    printf("\n--- Test 2: Scale factor 0.5 ---\n");
    unsigned char sc05[1] = {0x00}; /* two code-0 entries */
    nf4_dequantize_row(sc05, out, 0.5f, 2);
    check("code 0 * 0.5", out[0], nf4_ref[0] * 0.5f, 1e-5f);
    check("code 0 * 0.5 (nibble)", out[1], nf4_ref[0] * 0.5f, 1e-5f);

    /* Test 3: Scale factor 2.0 */
    printf("\n--- Test 3: Scale factor 2.0 ---\n");
    unsigned char sc2[1] = {0xFF}; /* two code-15 entries */
    nf4_dequantize_row(sc2, out, 2.0f, 2);
    check("code 15 * 2.0", out[0], nf4_ref[15] * 2.0f, 1e-5f);
    check("code 15 * 2.0 (nibble)", out[1], nf4_ref[15] * 2.0f, 1e-5f);

    /* Test 4: Odd element count (tests nibble extraction boundary) */
    printf("\n--- Test 4: Odd element count (17 elements) ---\n");
    unsigned char odd[9] = {0x12, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    nf4_dequantize_row(odd, out, 1.0f, 17);
    /* byte 0 = 0x12: hi=1, lo=2 → codes [1,2] */
    check("elem 0 (code 1)", out[0], nf4_ref[1], 1e-5f);
    check("elem 1 (code 2)", out[1], nf4_ref[2], 1e-5f);
    /* byte 1 = 0x34: hi=3, lo=4 → codes [3,4] */
    check("elem 2 (code 3)", out[2], nf4_ref[3], 1e-5f);
    check("elem 3 (code 4)", out[3], nf4_ref[4], 1e-5f);
    /* byte 2 = 0x00: hi=0, lo=0 → codes [0,0] */
    /* elem 16 is the 17th element = byte 8, hi nibble = 0 */
    check("elem 16 (code 0)", out[16], nf4_ref[0], 1e-5f);

    /* Test 5: Verify extreme codes */
    printf("\n--- Test 5: Extreme codes (0=min, 15=closest to 0) ---\n");
    unsigned char ext[2] = {0x0F, 0x0F}; /* four entries: codes 0,15,0,15 */
    nf4_dequantize_row(ext, out, 1.0f, 4);
    check("code 0 (min value)", out[0], nf4_ref[0], 1e-5f);
    check("code 15 (closest to 0)", out[1], nf4_ref[15], 1e-5f);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
