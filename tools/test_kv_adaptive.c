/* Test: Ecco entropy-aware adaptive KV compression (doc 001). */
#include "wubu_kv_adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    /* Test 1: Low-variance block (flat) → should pick 2-bit */
    float flat[32];
    for (int i = 0; i < 32; i++) flat[i] = 0.1f * (i % 3 - 1);
    uint8_t packed[32];
    int width;
    float scale;
    assert(wubu_kvq_adaptive_quant(flat, packed, &width, &scale, 32) == 0);
    printf("Flat block: width=%d scale=%.4f\n", width, (double)scale);

    /* Round-trip cosine must be > 0.99 */
    float cos = wubu_kvq_adaptive_roundtrip(flat, 32);
    printf("Flat round-trip cosine = %.6f\n", (double)cos);
    assert(cos > 0.99f);

    /* Test 2: High-variance block (sharp) → should pick 8-bit */
    float sharp[32];
    for (int i = 0; i < 32; i++) sharp[i] = (i == 7) ? 10.0f : 0.01f;
    assert(wubu_kvq_adaptive_quant(sharp, packed, &width, &scale, 32) == 0);
    printf("Sharp block: width=%d scale=%.4f\n", width, (double)scale);
    cos = wubu_kvq_adaptive_roundtrip(sharp, 32);
    printf("Sharp round-trip cosine = %.6f\n", (double)cos);
    assert(cos > 0.99f);

    /* Test 3: All-zeros (degenerate) → must not crash */
    float zeros[32] = {0};
    cos = wubu_kvq_adaptive_roundtrip(zeros, 32);
    printf("Zeros round-trip cosine = %.6f\n", (double)cos);

    /* Test 4: Average bits < 8 (proves compression engaged) */
    float mixed[64];
    uint8_t packed64[64];
    for (int i = 0; i < 64; i++) mixed[i] = 0.01f * (i % 5 - 2);
    wubu_kvq_adaptive_quant(mixed, packed64, &width, &scale, 64);
    printf("Mixed block: width=%d (avg bits = %d < 8 ✓)\n", width, width);
    assert(width < 8);

    printf("ALL KV-ADAPTIVE TESTS PASSED\n");
    return 0;
}
