/*
 * test_sys_tune.c -- L10/N06/N10/O03 verification.
 */
#include "wubu_sys_tune.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_sys_tune (L10/N06/N10/O03) ===\n");

    /* L10 SeerAttention: sharp head (e=0) keeps all; diffuse (e=1) keeps min. */
    CHECK(fabs(wubu_seer_keep_frac(0.0f, 0.1f) - 1.0f) < 1e-5f, "sharp -> keep all");
    CHECK(fabs(wubu_seer_keep_frac(1.0f, 0.1f) - 0.1f) < 1e-5f, "diffuse -> min_f");
    float mid = wubu_seer_keep_frac(0.5f, 0.1f);
    CHECK(mid > 0.1f && mid < 1.0f, "mid entropy -> between");

    /* N06 NUMA: >=1 always. */
    CHECK(wubu_numa_nodes() >= 1, "numa nodes >= 1");

    /* N10 energy: sum of terms, clamps negatives. */
    CHECK(fabs(wubu_energy_per_token(1.0, 2.0, 0.5) - 3.5) < 1e-9, "energy sums");
    CHECK(wubu_energy_per_token(-1.0, -2.0, -3.0) == 0.0, "negative energy -> 0");

    /* O03 tile factor: scales with sqrt(n), clamped. */
    int t1 = wubu_tile_factor(16, 1, 64);
    int t2 = wubu_tile_factor(1024, 1, 64);
    CHECK(t2 >= t1, "bigger n -> >= tile");
    CHECK(t1 >= 1 && t2 <= 64, "tile within [tmin,tmax]");
    CHECK(wubu_tile_factor(0, 1, 64) == 1, "n=0 -> tmin");

    if (failures == 0) { printf("ALL SYS-TUNE TESTS PASSED\n"); return 0; }
    printf("%d SYS-TUNE TEST(S) FAILED\n", failures);
    return 1;
}
