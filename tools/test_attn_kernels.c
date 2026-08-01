/*
 * test_attn_kernels.c -- P11/P15/O20 verification.
 */
#include "wubu_attn_kernels.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_attn_kernels (P11/P15/O20) ===\n");

    /* P11 int2 dequant: reconstruct 8 values from 2 bytes (4 levels/byte). */
    float orig[8] = {0.0f, 1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f};
    float scale = 1.0f, zero = 0.0f;
    unsigned char packed[2] = {0};
    for (int i = 0; i < 8; i++)            /* pack level = round(orig) */
        packed[i >> 2] |= ((unsigned char)(orig[i] + zero)) << (2 * (i & 3));
    float out[8];
    wubu_int2_dequant(packed, 8, scale, zero, out);
    int ok = 1; for (int i = 0; i < 8; i++) if (fabsf(out[i] - orig[i]) > 1e-5f) ok = 0;
    CHECK(ok, "int2 round-trip reconstructs 4-level values");
    CHECK(wubu_int2_dequant(NULL, 8, 1.0f, 0.0f, out) == 0, "null packed -> 0");

    /* P15 fused spec-verify. */
    CHECK(wubu_spec_verify_fused(0.51f, 0.50f, 0.02f) == 1, "close scores -> accept");
    CHECK(wubu_spec_verify_fused(0.7f, 0.5f, 0.02f) == 0, "far scores -> reject");

    /* O20 plasticity bits: high p -> bmax; low p -> bmin. */
    int hi = wubu_plasticity_bits(1.0f, 2, 8);
    int lo = wubu_plasticity_bits(0.0f, 2, 8);
    CHECK(hi == 8, "high plasticity -> bmax");
    CHECK(lo == 2, "low plasticity -> bmin");

    if (failures == 0) { printf("ALL ATTN-KERNELS TESTS PASSED\n"); return 0; }
    printf("%d ATTN-KERNELS TEST(S) FAILED\n", failures);
    return 1;
}
