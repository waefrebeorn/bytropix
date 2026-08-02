/* test_pim.c -- Theme IS batch 1: the PIM/near-memory frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_pim.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_pim (IS batch 1) ===\n");

    /* IS01: memory-bound GEMV over KV -> offload */
    CHECK(wubu_pim_offload(0, 100000, 100, 1000.0f, 1000.0f) == 1,
          "memory-bound offloads");
    CHECK(wubu_pim_offload(0, 100, 100000, 1000.0f, 1000.0f) == 0,
          "compute-bound stays");

    /* IS03: crossbar GEMV emulation */
    {
        float w[2][2] = { { 1, 2 }, { 3, 4 } }, v[2] = { 1, 1 }, out[2];
        CHECK(wubu_pim_gemv(&w[0][0], 2, 2, v, out, 8) == 0, "gemv runs");
        NEAR(out[0], 3.0f, 0.1f);
        NEAR(out[1], 7.0f, 0.1f);
    }

    /* IS04: quantized MAC with bit-cell clip */
    {
        int8_t w[2][2] = { { 127, 100 }, { -100, 50 } };
        int8_t v[2] = { 2, 1 };
        int32_t out[2];
        CHECK(wubu_pim_quant_gemv(&w[0][0], 2, 2, v, out, 4) == 0, "quant gemv");
        /* 127 clips to 7 (4-bit) */
        CHECK(out[0] == 7 * 2 + 7 * 1, "bit-cell clip applied");
    }

    /* IS05: tier costs */
    CHECK(wubu_pim_tier_cost(2, 1) < wubu_pim_tier_cost(4, 1), "RRAM cheaper");
    CHECK(wubu_pim_tier_cost(0, 1) > 0, "HBM positive");

    /* IS07: capacity wall */
    CHECK(wubu_pim_capacity(2048, 1024, 1.5f) == 1, "fits with margin");
    CHECK(wubu_pim_capacity(1024, 1024, 1.5f) == 0, "margin fails");

    /* IS10: bytes moved */
    CHECK(wubu_pim_bytes_moved(4, 4, 2) == 32, "4x4x2 bytes");

    /* IS09: hybrid dispatch */
    CHECK(wubu_pim_dispatch(1, 1000, 10, 0.5f, 0.9f) == 1, "PIM for GEMV");
    CHECK(wubu_pim_dispatch(0, 1000, 10, 0.9f, 0.5f) == 0, "NPU otherwise");

    /* IS12: channel-last layout */
    {
        float w[4] = { 1, 2, 3, 4 }, out[4];
        wubu_pim_layout(w, 2, 2, out);
        NEAR(out[0], 1.0f, 1e-6f); NEAR(out[1], 3.0f, 1e-6f);
        NEAR(out[2], 2.0f, 1e-6f); NEAR(out[3], 4.0f, 1e-6f);
    }

    /* IS13: analog noise */
    NEAR(wubu_pim_noise(1.0f, 8), 1.0f + 1.0f / 512.0f, 1e-4f);

    /* IS14: cost model */
    NEAR(wubu_pim_op_cost(0, 100, 10, 2.0f, 1.0f), 210.0f, 1e-4f);

    /* IS15: batching */
    {
        long ops[6] = { 10, 20, 30, 40, 50, 60 };
        CHECK(wubu_pim_batch(ops, 6, 100) == 2, "threshold-accumulated batches");
    }

    /* IS18: near-memory reduce */
    {
        float p[4] = { 1, 2, 3, 4 }, s = 0;
        CHECK(wubu_pim_reduce(p, 4, &s) == 0 && s == 10.0f, "reduced sum");
    }

    if (failures == 0) printf("ALL PIM TESTS PASSED\n");
    else printf("%d PIM FAILURES\n", failures);
    return failures ? 1 : 0;
}
