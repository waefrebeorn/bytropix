/* tools/test_heldout_eval.c
 *
 * Held-out validation for self-improvement (Track C2). Verifies the
 * reward-hacking guard:
 *   1. a genuine improvement passes (train ~ eval, both above floor)
 *   2. a mutation that fails held-out is rejected
 *   3. a mutation that games in-sample but collapses held-out is SUSPECT
 *   4. name mapping
 */
#include "wubu_heldout_eval.h"
#include <stdio.h>
#include <string.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

int main(void)
{
    printf("=== test_heldout_eval (reward-hacking guard) ===\n");
    const float floor = 0.5f, max_gap = 0.15f;

    /* 1. genuine improvement: train 0.82, eval 0.78, gap 0.04 -> pass */
    CHECK(wubu_heldout_eval_decide(0.82f, 0.78f, floor, max_gap) == WUBU_HELDOUT_PASS,
          "genuine improvement passes held-out");

    /* 2. held-out below floor -> fail (don't promote) */
    CHECK(wubu_heldout_eval_decide(0.90f, 0.40f, floor, max_gap) == WUBU_HELDOUT_FAIL,
          "held-out below floor -> fail");

    /* 3. gaming signature: train 0.95, eval 0.60, gap 0.35 > 0.15 -> suspect */
    CHECK(wubu_heldout_eval_decide(0.95f, 0.60f, floor, max_gap) == WUBU_HELDOUT_SUSPECT,
          "in-sample >> held-out -> SUSPECT (reward hacking)");

    /* 4. a gap within tolerance is fine (not suspect) */
    CHECK(wubu_heldout_eval_decide(0.80f, 0.70f, floor, max_gap) == WUBU_HELDOUT_PASS,
          "gap within tolerance -> pass");

    /* 5. names */
    CHECK(strcmp(wubu_heldout_eval_name(WUBU_HELDOUT_SUSPECT), "suspect") == 0,
          "suspect name");
    CHECK(strcmp(wubu_heldout_eval_name(WUBU_HELDOUT_PASS), "pass") == 0,
          "pass name");

    printf("=== test_heldout_eval: %d checks, %d failures ===\n", checks, fails);
    if (fails) return 1;
    printf("=== ALL HELDOUT_EVAL TESTS PASSED (no reward hacking) ===\n");
    return 0;
}
