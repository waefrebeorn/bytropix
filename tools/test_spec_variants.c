/*
 * test_spec_variants.c -- M11/M13/M14/L14 verification.
 */
#include "wubu_spec_variants.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_spec_variants (M11/M13/M14/L14) ===\n");

    /* M13 co-design: high acceptance -> large K; KV-bound (b*<1) -> lo bits. */
    int K, bits;
    wubu_spec_kv_codesign(0.9f, 50.0, 8, 2, 16, &K, &bits);
    CHECK(K > 1, "high acceptance -> K>1");
    CHECK(bits == 16, "weight-bound -> hi bits");
    wubu_spec_kv_codesign(0.9f, 0.3, 8, 2, 16, &K, &bits);
    CHECK(bits == 2, "KV-bound -> lo bits");

    /* M14 blockwise verify: K=10, nb=4 -> 3 blocks. */
    CHECK(wubu_blockwise_verify_blocks(10, 4) == 3, "ceil(10/4)=3 blocks");
    CHECK(wubu_blockwise_verify_blocks(0, 4) == 0, "K=0 -> 0 blocks");

    /* M11 KV reuse: prefix longer than pos -> reuse ok. */
    CHECK(wubu_kv_reuse_ok(5, 8) == 1, "prefix>=pos -> reuse");
    CHECK(wubu_kv_reuse_ok(9, 8) == 0, "prefix<pos -> no reuse");

    /* L14 offload: cold (low r*m) -> offload; hot -> keep. */
    CHECK(wubu_offload_decision(0.1f, 0.1f, 0.5f) == 1, "cold -> offload");
    CHECK(wubu_offload_decision(0.9f, 0.9f, 0.5f) == 0, "hot -> keep");

    if (failures == 0) { printf("ALL SPEC-VARIANTS TESTS PASSED\n"); return 0; }
    printf("%d SPEC-VARIANTS TEST(S) FAILED\n", failures);
    return 1;
}
