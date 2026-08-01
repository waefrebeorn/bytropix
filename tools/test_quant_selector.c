/*
 * test_quant_selector.c -- N04/N05/N09 verification.
 */
#include "wubu_quant_selector.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_quant_selector (N04/N05/N09) ===\n");

    /* N04 batch-aware: small batch (<B*) -> weight-bound -> hi precision. */
    int bw, bkv;
    wubu_batch_quant(1, 30.0, 4, 16, 2, 16, &bw, &bkv);
    CHECK(bw == 16 && bkv == 16, "small batch -> hi precision (weight-bound)");
    wubu_batch_quant(100, 30.0, 4, 16, 2, 16, &bw, &bkv);
    CHECK(bkv == 2, "large batch (>B*) -> KV compressed (KV-bound)");

    /* N05 context ladder: short -> hi, long -> lo, monotonic non-increasing. */
    int b0 = wubu_ctx_precision_ladder(1000, 100000, 2, 16);
    int b1 = wubu_ctx_precision_ladder(50000, 100000, 2, 16);
    int b2 = wubu_ctx_precision_ladder(100000, 100000, 2, 16);
    CHECK(b0 > b1 && b1 >= b2, "precision decreases with context");
    CHECK(b0 == 16, "short ctx -> b_hi");
    CHECK(b2 == 2, "full ctx -> b_lo");

    /* N09 PMC roofline: 1e9 bytes, 1e9 cycles, 2e9 Hz -> 2 GB/s. */
    double r = wubu_pmc_roofline(1e9, 1e9, 2e9);
    CHECK(fabs(r - 2e9) < 1e6, "PMC roofline = 2 GB/s");
    CHECK(wubu_pmc_roofline(0, 1e9, 2e9) < 0.0, "bad input -> -1");

    if (failures == 0) { printf("ALL QUANT-SELECTOR TESTS PASSED\n"); return 0; }
    printf("%d QUANT-SELECTOR TEST(S) FAILED\n", failures);
    return 1;
}
