/*
 * test_kv_compress.c -- L07/L09 verification.
 */
#include "wubu_kv_compress.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv_compress (L07/L09) ===\n");

    /* scores: peaky at slot 3 and 7 (heavy hitters). */
    float s[10];
    for (int i = 0; i < 10; i++) s[i] = 0.1f;
    s[3] = 0.9f; s[7] = 0.8f;

    /* L09 keep top 30% (3 of 10) -> retains 3,7, and the next highest (0.1). */
    int out[10];
    int k = wubu_kv_keep_top_score(s, 10, 0.3f, out);
    CHECK(k == 3, "keep 3 of 10 at 0.3");
    int has3 = 0, has7 = 0;
    for (int i = 0; i < k; i++) { if (out[i]==3) has3=1; if (out[i]==7) has7=1; }
    CHECK(has3 && has7, "heavy hitters (3,7) retained");
    CHECK(wubu_kv_keep_top_score(s, 10, 1.0f, out) == 10, "frac>=1 keeps all");
    CHECK(wubu_kv_keep_top_score(s, 10, 0.0f, out) == 0, "frac<=0 keeps none");
    CHECK(wubu_kv_keep_top_score(NULL, 10, 0.3f, out) == 0, "null scores -> 0");

    /* L07 SnapKV clusters: 10 slots, 5 clusters of 2. Peak cluster (3,4) mean
     * ~0.5, (6,7) mean ~0.45 -> keep those two clusters => 4 slots. */
    int out2[10];
    int kc = wubu_kv_keep_clusters(s, 10, 5, 2, out2);
    CHECK(kc == 4, "keep 2 of 5 clusters => 4 slots");
    int has3b = 0, has7b = 0;
    for (int i = 0; i < kc; i++) { if (out2[i]==3||out2[i]==4) has3b=1; if (out2[i]==6||out2[i]==7) has7b=1; }
    CHECK(has3b && has7b, "high-attention clusters retained");

    /* L08 PyramidKV: shallow layer keeps more than deep. */
    float shallow = wubu_pyramid_keep(0.5f, 0.0f, 2.0f);
    float deep    = wubu_pyramid_keep(0.5f, 1.0f, 2.0f);
    CHECK(shallow > deep, "shallow layer keeps more KV than deep");
    CHECK(shallow <= 1.0f && deep >= 0.0f, "pyramid keep within [0,1]");

    if (failures == 0) { printf("ALL KV-COMPRESS TESTS PASSED\n"); return 0; }
    printf("%d KV-COMPRESS TEST(S) FAILED\n", failures);
    return 1;
}
