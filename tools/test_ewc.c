/* test_ewc.c -- PATHS #321 continual learning without catastrophic forgetting.
 * Verifies the existing wubu_ewc module: Fisher importance estimation, EWC
 * penalty grows with drift from anchor on important dims, stable-dim
 * protection, so learning a new task resists overwriting important weights. */
#include "wubu_ewc.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    double params[WUBU_EWC_DIMS] = {0};
    for (int i = 0; i < WUBU_EWC_DIMS; i++) params[i] = 0.5;
    wubu_ewc_t e;
    CHECK(wubu_ewc_init(&e, params, WUBU_EWC_DIMS, 1.0) == 0, "init with anchor ok");
    CHECK(e.initialized == 1, "initialized");

    /* estimate Fisher importance: dims 0,1,2 important (big gradients) */
    double grads[WUBU_EWC_DIMS] = {0};
    grads[0] = 10.0; grads[1] = 8.0; grads[2] = 5.0;   /* important */
    CHECK(wubu_ewc_estimate_fisher(&e, grads, WUBU_EWC_DIMS) == 0, "estimate fisher ok");
    CHECK(e.fisher[0] > e.fisher[10], "dim 0 more important than dim 10");
    CHECK(wubu_ewc_is_stable(&e, 0) == 1, "dim 0 stable (protected)");

    /* penalty is 0 at the anchor (no drift) */
    CHECK(NEAR(wubu_ewc_penalty(&e, params, WUBU_EWC_DIMS), 0.0, 1e-9), "penalty 0 at anchor");

    /* drift on an IMPORTANT dim -> large penalty (resists change) */
    double drifted_important[WUBU_EWC_DIMS];
    for (int i = 0; i < WUBU_EWC_DIMS; i++) drifted_important[i] = params[i];
    drifted_important[0] += 1.0;   /* big drift on important dim 0 */
    double pen_imp = wubu_ewc_penalty(&e, drifted_important, WUBU_EWC_DIMS);

    /* same drift on an UNIMPORTANT dim -> small penalty (free to change) */
    double drifted_unimportant[WUBU_EWC_DIMS];
    for (int i = 0; i < WUBU_EWC_DIMS; i++) drifted_unimportant[i] = params[i];
    drifted_unimportant[10] += 1.0;   /* same drift on unimportant dim 10 */
    double pen_unimp = wubu_ewc_penalty(&e, drifted_unimportant, WUBU_EWC_DIMS);

    CHECK(pen_imp > pen_unimp, "drift on important dim penalized more than unimportant");
    CHECK(pen_imp > 0, "important-dim drift has positive penalty");

    /* stable count: the 3 important dims are protected */
    int stable = wubu_ewc_stable_count(&e, 1.0);
    CHECK(stable >= 3, "at least 3 stable (protected) dims");

    /* the EWC penalty is what PREVENTS catastrophic forgetting: a gradient
     * step on an important weight is resisted, so task-B learning doesn't
     * overwrite task-A's important weights. The penalty magnitude encodes
     * that resistance. */
    CHECK(pen_imp > 1.0, "important-dim penalty exceeds lambda (strong protection)");

    printf("\n=== test_ewc: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
