/*
 * test_eval_qat.c -- Z01-Z05 + AA01-AA04 verification.
 */
#include "wubu_eval_qat.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_eval_qat (Z01-Z05/AA01-AA04) ===\n");

    /* Z01 NIAH inject: len 100, 3 needles -> 3 distinct positions. */
    int pos[3];
    int np = wubu_niah_inject(100, 3, pos);
    CHECK(np == 3, "3 needle positions");
    CHECK(pos[0]!=pos[1] && pos[1]!=pos[2] && pos[0]!=pos[2], "positions distinct");

    /* Z02 RULER retrieve: keys [10,20,30] vals [1,2,3], query 20 -> 2. */
    int k[3] = {10,20,30}, v[3] = {1,2,3}, out;
    CHECK(wubu_ruler_retrieve(k, v, 3, 20, &out)==1 && out==2, "retrieve key20->2");

    /* Z03 RULER multi-hop: chain 1->2->3->4, start 1 depth 3 -> 4. */
    int kk[3] = {1,2,3}, nx[3] = {2,3,4}, mh;
    CHECK(wubu_ruler_multihop(kk, nx, 3, 1, 3, &mh)==1 && mh==4, "multihop 1->4 depth3");

    /* Z04 RULER aggregate: ctx [5,5,7,5], target 5 -> 3. */
    int ctx[4] = {5,5,7,5};
    CHECK(wubu_ruler_aggregate(ctx, 4, 5) == 3, "aggregate count 5 -> 3");

    /* Z05 haystack gen: len 10 -> 10 noise tokens (nonzero). */
    int toks[10];
    int ng = wubu_haystack_gen(10, toks);
    CHECK(ng==10 && toks[0]!=0, "haystack 10 tokens");

    /* AA01 fake-quant: x=0.37, step=0.25 -> round(1.48)=1 -> 0.25. */
    CHECK(fabsf(wubu_fakequant(0.37f,0.25f,-1,1) - 0.25f) < 1e-5f, "fakequant 0.37->0.25");

    /* AA02 QAT STE: x=0.3 in [-1,1] -> q=0.25, grad passes. */
    float q; int gp;
    wubu_qat_ste(0.3f, 0.25f, -1.0f, 1.0f, &q, &gp);
    CHECK(fabsf(q-0.25f)<1e-5f && gp==1, "QAT STE q=0.25 grad pass");

    /* AA03 per-channel dequant: q=7, scale=0.5, zero=3 -> (7-3)*0.5=2.0. */
    CHECK(fabsf(wubu_dequant_pc(7, 0.5f, 3) - 2.0f) < 1e-5f, "dequant (7-3)*0.5=2");

    /* AA04 noise inject: deterministic for same seed, within amp. */
    float a = wubu_noise_inject(1.0f, 42, 0.1f);
    float b = wubu_noise_inject(1.0f, 42, 0.1f);
    CHECK(fabsf(a-b)<1e-6f && fabsf(a-1.0f)<=0.1f+1e-5f, "noise deterministic, bounded");

    if (failures == 0) { printf("ALL EVAL-QAT TESTS PASSED\n"); return 0; }
    printf("%d EVAL-QAT TEST(S) FAILED\n", failures);
    return 1;
}
