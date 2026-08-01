/*
 * test_parallel_spec.c -- V01-V04 + W01-W03 verification.
 */
#include "wubu_parallel_spec.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_parallel_spec (V01-V04/W01-W03) ===\n");

    /* V01 EAGLE-3: feat scores [0.1,0.9,0.5] -> drafted id 1. */
    float fs[3] = {0.1f,0.9f,0.5f};
    int d;
    CHECK(wubu_eagle3_draft(fs, 3, &d) == 1 && d == 1, "EAGLE-3 drafts argmax id 1");

    /* V02 P-EAGLE: K=3 drafts [10,11,12], match {1,0,1} -> accept {10,12}. */
    int drafts[3] = {10,11,12};
    char m[3] = {1,0,1};
    int acc[3]; int na = wubu_peagle_verify(drafts, m, 3, acc);
    CHECK(na == 2 && acc[0]==10 && acc[1]==12, "P-EAGLE accepts {10,12}");

    /* V03 tree parents: n=4 -> [-1,0,1,2]. */
    int par[4];
    wubu_tree_attn_parents(4, par);
    CHECK(par[0]==-1 && par[1]==0 && par[2]==1 && par[3]==2, "tree parents chain");

    /* V04 Kangaroo: shallow match only -> accept. */
    CHECK(wubu_kangaroo_accept(1, 0) == 1, "Kangaroo shallow match -> accept");
    CHECK(wubu_kangaroo_accept(0, 0) == 0, "Kangaroo no match -> reject");

    /* W01 NoPE enabled. */
    CHECK(wubu_nope_enabled() == 1, "NoPE flag on");

    /* W02 ALiBi bias: n=3,d=1,slope=0.5 -> bias[1,0]=-0.5, bias[2,0]=-1.0, bias[2,1]=-0.5. */
    float bias[9];
    wubu_alibi_bias(bias, 3, 1, 0.5f);
    CHECK(fabsf(bias[3]-(-0.5f))<1e-5f, "alibi[1,0]=-0.5"); /* (1*3+0)=3 */
    CHECK(fabsf(bias[6]-(-1.0f))<1e-5f, "alibi[2,0]=-1.0"); /* (2*3+0)=6 */
    CHECK(fabsf(bias[7]-(-0.5f))<1e-5f, "alibi[2,1]=-0.5"); /* (2*3+1)=7 */

    /* W03 FFN-first flag. */
    CHECK(wubu_ffn_first_enabled() == 1, "FFN-first flag on");

    if (failures == 0) { printf("ALL PARALLEL-SPEC TESTS PASSED\n"); return 0; }
    printf("%d PARALLEL-SPEC TEST(S) FAILED\n", failures);
    return 1;
}
