/* Test: wubu_yarn (Round-3 #241 — YaRN NTK-aware extrapolation). */
#include "wubu_yarn.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void) {
    int d = 64;
    double scale[32], ramp[32];
    /* Extend 262144 -> 1010000 context (Qwen3.6). */
    int rc = wubu_yarn_scales(d, 262144.0, 1010000.0, 16.0, scale, ramp);
    assert(rc == 0);
    /* High-freq dims (small i) should be ~1 (little scaling). */
    printf("YaRN scale[0] (high-freq) = %.4f (expect ~1)\n", scale[0]);
    assert(fabs(scale[0] - 1.0) < 0.05);
    /* Low-freq dims (large i) should be > 1 (extrapolated). */
    printf("YaRN scale[31] (low-freq) = %.4f (expect > 1)\n", scale[31]);
    assert(scale[31] > 1.0);
    /* Ramp monotonic 0->1. */
    assert(ramp[0] < ramp[31]);
    /* Apply: angle scales by the per-dim factor. */
    double theta_out;
    wubu_yarn_apply(3.14159, scale, 32, 0, &theta_out);
    printf("YaRN apply theta: %.4f -> %.4f\n", 3.14159, theta_out);
    assert(fabs(theta_out - 3.14159*scale[0]) < 1e-9);
    /* Bad args. */
    assert(wubu_yarn_scales(d, 100.0, 50.0, 16.0, scale, ramp) == -1); /* target<train */
    printf("ALL YARN TESTS PASSED\n");
    return 0;
}
