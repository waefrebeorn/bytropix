/* test_hopfield.c -- Theme IL: modern Hopfield / associative memory. */
#include <stdio.h>
#include <math.h>
#include "wubu_hopfield.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_hopfield (IL01-IL07) ===\n");

    /* IL01: retrieval -- a stored pattern retrieved from itself at a
     * sharp beta reconstructs a pattern close to it */
    {
        /* two orthogonal-ish patterns in 4-d */
        float X[8] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f
        };
        float xi[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
        float out[4];
        CHECK(wubu_hopfield_retrieve(X, 2, 4, xi, 8.0f, out) == 0, "retrieve");
        CHECK(out[0] > 0.9f && out[1] < 0.1f, "retrieves the first pattern");
        /* a cue closer to pattern 2 retrieves pattern 2 */
        float xi2[4] = { 0.1f, 1.0f, 0.0f, 0.0f };
        wubu_hopfield_retrieve(X, 2, 4, xi2, 8.0f, out);
        CHECK(out[1] > 0.9f && out[0] < 0.1f, "retrieves the second pattern");
        /* null check */
        CHECK(wubu_hopfield_retrieve(NULL, 2, 4, xi, 8.0f, out) == -1, "null rejected");
    }

    /* IL02: attention beta = 1/sqrt(d) */
    NEAR(wubu_hopfield_beta_attention(4), 0.5f, 1e-6f);
    NEAR(wubu_hopfield_beta_attention(16), 0.25f, 1e-6f);
    NEAR(wubu_hopfield_beta_attention(0), 1.0f, 1e-6f);

    /* IL03: exponential capacity */
    {
        float c8 = wubu_hopfield_capacity(8, 0.5f);
        float c16 = wubu_hopfield_capacity(16, 0.5f);
        CHECK(c16 > c8 * c8 * 0.9f, "capacity ~ exponential in dim");
        NEAR(wubu_hopfield_capacity(0, 0.5f), 1.0f, 1e-6f);
    }

    /* IL04: denoise -- a corrupted cue converges back to the pattern */
    {
        float X[8] = {
            1.0f, 1.0f, 1.0f, 1.0f,
            -1.0f, -1.0f, -1.0f, -1.0f
        };
        float cue[4] = { 0.8f, 0.9f, 0.7f, -0.4f };  /* pattern 1 w/ noise */
        float out[4];
        int it = wubu_hopfield_denoise(X, 2, 4, cue, 6.0f, 1e-4f, 32, out);
        CHECK(it >= 0 && out[0] > 0.9f && out[3] > 0.0f,
              "denoise converges to the positive pattern");
        CHECK(wubu_hopfield_denoise(X, 2, 4, cue, 6.0f, 1e-4f, 32, NULL) == -1,
              "null out rejected");
    }

    /* IL05: decay */
    NEAR(wubu_hopfield_decay(1.0f, 0, 10.0f), 1.0f, 1e-6f);
    NEAR(wubu_hopfield_decay(1.0f, 10, 10.0f), 0.5f, 1e-4f);
    NEAR(wubu_hopfield_decay(1.0f, 20, 10.0f), 0.25f, 1e-4f);
    NEAR(wubu_hopfield_decay(1.0f, 5, 0.0f), 0.0f, 1e-6f);

    /* IL06: consolidation */
    NEAR(wubu_hopfield_consolidate(1.0f, 0.5f, 2.0f), 2.0f, 1e-6f);
    NEAR(wubu_hopfield_consolidate(1.0f, -5.0f, 2.0f), 1.0f, 1e-6f);
    NEAR(wubu_hopfield_consolidate(1.0f, 0.5f, -1.0f), 1.0f, 1e-6f);

    /* IL07: topk by overlap */
    {
        float X[12] = {  /* 3 patterns, 4-d */
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        float xi[4] = { 0.0f, 0.9f, 0.2f, 0.0f };
        int idx[3];
        int n = wubu_hopfield_topk(X, 3, 4, xi, 2, idx);
        CHECK(n == 2 && idx[0] == 1 && idx[1] == 2, "topk: pattern 1 then 2");
        n = wubu_hopfield_topk(X, 3, 4, xi, 5, idx);
        CHECK(n == 3, "topk capped at the pattern count");
        CHECK(wubu_hopfield_topk(X, 3, 4, xi, 0, idx) == -1, "k=0 rejected");
    }

    if (failures == 0) printf("ALL HOPFIELD TESTS PASSED\n");
    else printf("%d HOPFIELD FAILURES\n", failures);
    return failures ? 1 : 0;
}
