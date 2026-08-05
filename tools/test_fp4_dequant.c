/*
 * test_fp4_dequant.c — verify MXFP4 and NVFP4 dequantization against
 * known test vectors.  Build:
 *   gcc -O2 -std=c11 -I include -I include/win32 -DWUBU_BUILD_WIN \
 *       -include wubu_win.h -o test_fp4_dequant tools/test_fp4_dequant.c \
 *       src/wubu_dequant_fp4.c
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "wubu_dequant_fp4.h"

static int tests_run = 0, tests_passed = 0;

static void check(const char *name, float got, float expected, float tol) {
    tests_run++;
    /* Handle ±Inf and NaN specially */
    if (isinf(expected)) {
        if (isinf(got) && (got > 0) == (expected > 0)) {
            tests_passed++;
            printf("  PASS %s: got=%.6f expected=%.6f\n", name, got, expected);
            return;
        }
    } else if (isnan(expected)) {
        if (isnan(got)) { tests_passed++; printf("  PASS %s (NaN)\n", name); return; }
    } else if (fabsf(got - expected) <= tol) {
        tests_passed++;
        printf("  PASS %s: got=%.6f expected=%.6f\n", name, got, expected);
    } else {
        printf("  FAIL %s: got=%.6f expected=%.6f (tol=%.6f)\n", name, got, expected, tol);
    }
}

int main(void) {
    /* ---- MXFP4 test ---- */
    /* One block: scale=0xFF (E8M0 → 2^128), values [3,2,1,0] = [2,1,0.5,0]
       With scale 2^128 these would overflow float, so use scale=0x80 → 2^1 = 2.0 */
    {
        /* scale byte = 0x80 → E=128 → scale = 2^(128-127) = 2.0 */
        /* qs: byte 0 = 0x32 (nibbles 3,2) → values 2.0, 1.0  → ×2.0 = 4.0, 2.0 */
        /*     byte 1 = 0x10 (nibbles 1,0) → values 0.5, 0.0  → ×2.0 = 1.0, 0.0 */
        /* Remaining bytes 0xFF, rest zeros — elements 4..31 are unused (only test 4) */
        unsigned char mxfp4_block[17];
        memset(mxfp4_block, 0, 17);
        mxfp4_block[0] = 0x80;          /* E8M0 scale = 2^1 = 2.0 */
        mxfp4_block[1] = 0x32;          /* nibbles: 3,2 → E2M1 2.0,1.0 */
        mxfp4_block[2] = 0x10;          /* nibbles: 1,0 → E2M1 0.5,0.0 */

        float out[32];
        memset(out, 0xFF, sizeof(out));
        dequantize_row_mxfp4(mxfp4_block, out, 4);

        check("MXFP4 elem0", out[0], 4.0f, 0.001f);  /* 2.0 × 2.0 */
        check("MXFP4 elem1", out[1], 2.0f, 0.001f);  /* 2.0 × 1.0 */
        check("MXFP4 elem2", out[2], 1.0f, 0.001f);  /* 2.0 × 0.5 */
        check("MXFP4 elem3", out[3], 0.0f, 0.001f);  /* 2.0 × 0.0 */

        /* Verify raw size */
        long sz = wubu_fp4_raw_size(39, 32);
        check("MXFP4 raw_size(32)", (float)sz, 17.0f, 0.0f);

        sz = wubu_fp4_raw_size(39, 64);
        check("MXFP4 raw_size(64)", (float)sz, 34.0f, 0.0f);
    }

    /* ---- NVFP4 test ---- */
    /* One block: 64 elements, 4 UE4M3 scales (one per 16-element sub-block) */
    /* Sub-block 0 scale = 0x78 → E=15, M=0 → (1+0/8)*2^(15-7) = 256.0
       Values: 0x32 → [2,1], 0x10 → [0.5, 0] → ×256 = 512, 256, 128, 0 */
    {
        unsigned char nvfp4_block[36];
        memset(nvfp4_block, 0, 36);
        /* scales (4 bytes UE4M3) */
        nvfp4_block[0] = 0x78;  /* sub-block 0: E=15,M=0 → 256.0 */
        nvfp4_block[1] = 0x00;  /* sub-block 1: 0.0 → all zeros */
        nvfp4_block[2] = 0x00;  /* sub-block 2: 0.0 */
        nvfp4_block[3] = 0x00;  /* sub-block 3: 0.0 */
        /* packed 4-bit values (32 bytes = 64 nibbles) */
        nvfp4_block[4]  = 0x32;  /* elems 0,1: 2.0, 1.0 → ×256 = 512, 256 */
        nvfp4_block[5]  = 0x10;  /* elems 2,3: 0.5, 0.0 → ×256 = 128, 0 */
        /* rest of values are 0 → zeros */

        float out[64];
        memset(out, 0xFF, sizeof(out));
        dequantize_row_nvfp4(nvfp4_block, out, 4);

        check("NVFP4 elem0", out[0], 512.0f, 0.001f);  /* 256 × 2.0 */
        check("NVFP4 elem1", out[1], 256.0f, 0.001f);  /* 256 × 1.0 */
        check("NVFP4 elem2", out[2], 128.0f, 0.001f);  /* 256 × 0.5 */
        check("NVFP4 elem3", out[3], 0.0f, 0.001f);    /* 256 × 0.0 */

        /* Verify raw size: 64 elements → 1 block × 36 bytes */
        long sz = wubu_fp4_raw_size(40, 64);
        check("NVFP4 raw_size(64)", (float)sz, 36.0f, 0.0f);

        /* 128 elements → 2 blocks × 36 = 72 */
        sz = wubu_fp4_raw_size(40, 128);
        check("NVFP4 raw_size(128)", (float)sz, 72.0f, 0.0f);

        /* 48 elements → 1 block (partial) × 36 = 36 */
        sz = wubu_fp4_raw_size(40, 48);
        check("NVFP4 raw_size(48)", (float)sz, 36.0f, 0.0f);
    }

    /* ---- E8M0 edge cases ---- */
    {
        /* scale=0 → 2^(-127) ≈ 0, all values should be ~0 */
        unsigned char block[17];
        memset(block, 0, 17);
        block[1] = 0x3F;  /* all 4-bit values = 3 (max) */
        float out[4];
        dequantize_row_mxfp4(block, out, 4);
        /* scale = 2^(-127) ≈ 5.88e-39, × 2.0 ≈ 1.17e-38 */
        check("MXFP4 scale=0 elem0", out[0], 0.0f, 1e-37f);

        /* scale=255 → 2^(128) — Inf in float, should not NaN */
        memset(block, 0, 17);
        block[0] = 255;
        block[1] = 0x11;  /* values 1,1 → 0.5, 0.5 */
        dequantize_row_mxfp4(block, out, 2);
        check("MXFP4 scale=255 elem0", out[0], INFINITY, 0.0f);
    }

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
