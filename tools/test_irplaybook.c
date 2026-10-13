/* test_irplaybook.c -- AGIEDR #307 agent incident-response playbook.
 * Proves: kill->contain->rollback order, fail-closed halt on failed step,
 * resolved flag, reset, fail-closed on NULL. */
#include "wubu_irplaybook.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_irplaybook_t p;
    CHECK(wubu_ir_init(&p) == 0, "init ok (idle)");
    CHECK(wubu_ir_state(&p) == WUBU_IR_IDLE, "starts idle");
    CHECK(wubu_ir_resolved(&p) == 0, "not resolved at start");

    /* run the runbook: kill -> contain -> rollback */
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_KILLED, "step 1: kill");
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_CONTAINED, "step 2: contain");
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_ROLLED_BACK, "step 3: rollback");
    CHECK(wubu_ir_resolved(&p) == 1, "resolved after rollback");
    CHECK(p.n_actions == 3, "3 runbook actions executed");

    /* further steps are no-ops once resolved */
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_ROLLED_BACK, "already resolved -> stays resolved");
    CHECK(p.n_actions == 3, "no extra actions after resolution");

    /* reset for next incident */
    CHECK(wubu_ir_reset(&p) == 0, "reset ok");
    CHECK(wubu_ir_state(&p) == WUBU_IR_IDLE, "back to idle");

    /* fail-closed: a failed step halts the runbook (no half-response) */
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_KILLED, "step 1: kill ok");
    CHECK(wubu_ir_step(&p, 0) == WUBU_IR_KILLED, "step 2 FAILS -> halts (stays killed)");
    CHECK(wubu_ir_resolved(&p) == 0, "not resolved (contain failed)");
    CHECK(p.n_actions == 1, "only 1 action before the failure");
    /* a subsequent successful step can retry from where it halted */
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_CONTAINED, "retry step 2: contain ok");
    CHECK(wubu_ir_step(&p, 1) == WUBU_IR_ROLLED_BACK, "step 3: rollback ok");
    CHECK(wubu_ir_resolved(&p) == 1, "resolved after successful retry");

    /* fail-closed on NULL */
    CHECK(wubu_ir_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_ir_step(NULL, 1) == WUBU_IR_IDLE, "NULL step -> idle");
    CHECK(wubu_ir_state(NULL) == WUBU_IR_IDLE, "NULL state -> idle");
    CHECK(wubu_ir_resolved(NULL) == 0, "NULL resolved -> 0");
    CHECK(wubu_ir_reset(NULL) == -1, "NULL reset -> -1");

    printf("\n=== test_irplaybook: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
