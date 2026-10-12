/* test_embedding_norm.c — BRAIN #238: embedding norm forward verified.
 *
 * Measured assertions on wubu_embed_scale_factor + wubu_embed_scale:
 *   1. factor == sqrt(d): sqrt(4)==2, sqrt(9)==3, sqrt(1)==1.
 *   2. factor(0)==0 (invalid).
 *   3. scaling each element by exactly sqrt(d).
 *   4. a unit row scaled by sqrt(d) has norm == d (var balanced to ~1/d -> sqrt).
 *   5. bad args rejected.
 */
#include "wubu_embedding_norm.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static int near(double a, double b, double eps) { return fabs(a - b) <= eps; }

int main(void) {
    printf("=== test_embedding_norm (embedding scaling, #238) ===\n");

    /* 1: sqrt(d). */
    T(near(wubu_embed_scale_factor(4), 2.0, 1e-9), "factor(4)==2");
    T(near(wubu_embed_scale_factor(9), 3.0, 1e-9), "factor(9)==3");
    T(near(wubu_embed_scale_factor(1), 1.0, 1e-9), "factor(1)==1");

    /* 2: invalid. */
    T(wubu_embed_scale_factor(0) == 0.0, "factor(0)==0");

    /* 3: scale by sqrt(d) exactly. d=4 -> *2. */
    float e[8] = {1,2,3,4, 5,6,7,8};   /* 2 rows x 4 dims */
    T(wubu_embed_scale(e, 2, 4) == 0, "scale ok");
    T(e[0]==2 && e[3]==8 && e[4]==10 && e[7]==16, "each element *sqrt(4)==*2");

    /* 4: unit row (all 1/sqrt(d)) scaled by sqrt(d) -> all 1, norm==sqrt(d). */
    int d = 4;
    float row[4];
    for (int i = 0; i < d; i++) row[i] = 1.0f / sqrtf((float)d);
    wubu_embed_scale(row, 1, d);
    double norm = 0.0;
    int ones = 1;
    for (int i = 0; i < d; i++) { norm += (double)row[i]*row[i]; if (!near(row[i], 1.0, 1e-5)) ones = 0; }
    T(ones, "unit row scaled -> all ones");
    T(near(sqrt(norm), sqrt((double)d), 1e-5), "scaled row norm == sqrt(d)");

    /* 5: bad args. */
    T(wubu_embed_scale(NULL, 2, 4) == -1, "NULL rejected");
    T(wubu_embed_scale(e, 0, 4) == -1, "n<=0 rejected");
    T(wubu_embed_scale(e, 2, 0) == -1, "d<=0 rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
