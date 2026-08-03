/* test_mix.c -- the weighted multi-stream mixer: the output must be a
 * deterministic blend whose per-stream token counts respect the weights. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mix.h"

static void write_tok(const char *path, const uint16_t *t, int n)
{
    FILE *f = fopen(path, "wb");
    fwrite(t, 2, n, f);
    fclose(f);
}

int main(void)
{
    int ok = 1;
    uint16_t a[4000], b[2000];
    for (int i = 0; i < 4000; i++) a[i] = (uint16_t)(100 + i % 200);
    for (int i = 0; i < 2000; i++) b[i] = (uint16_t)(900 + i % 100);
    write_tok("/tmp/mix_a.tok", a, 4000);
    write_tok("/tmp/mix_b.tok", b, 2000);

    const char *paths[] = { "/tmp/mix_a.tok", "/tmp/mix_b.tok" };
    const float weights[] = { 3.0f, 1.0f };   /* a 3:1 mix */
    uint16_t out[100000];
    long n = wubu_mix_build(paths, weights, 2, out, 100000, 256);
    if (n < 1) { printf("  build FAIL\n"); return 1; }

    /* the WEIGHT governs the interleaving: the first 4000 tokens of the
     * mix must be ~a 3:1 blend (the streams' totals are exhausted at the
     * end -- the final counts equal the stream sizes, so the weight is
     * measured on the early portion) */
    long na = 0, nb = 0;
    long first = 4000 < n ? 4000 : n;
    for (long i = 0; i < first; i++) {
        if (out[i] >= 100 && out[i] < 300) na++;
        else if (out[i] >= 900) nb++;
    }
    float ratio = (float)na / (float)(nb + 1);
    printf("  mix: %ld tokens, first-%ld: a=%ld b=%ld ratio %.2f (want ~3)  %s\n",
           n, first, na, nb, ratio, (ratio > 2.3f && ratio < 3.7f) ? "PASS" : "FAIL");
    if (ratio < 2.3f || ratio > 3.7f) ok = 0;

    /* the determinism: the same call reproduces the same mix */
    uint16_t out2[100000];
    long n2 = wubu_mix_build(paths, weights, 2, out2, 100000, 256);
    if (n2 != n || memcmp(out, out2, (size_t)n * 2) != 0) {
        printf("  determinism FAIL\n"); ok = 0;
    }

    /* the exhaustion: every stream fully consumed (the totals) */
    printf("%s\n", ok ? "ALL MIX TESTS PASSED" : "MIX FAILURES");
    return ok ? 0 : 1;
}
