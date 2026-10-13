/* wubu_irplaybook.c -- AGIEDR #307 agent incident-response playbook.
 * See wubu_irplaybook.h. */
#include "wubu_irplaybook.h"

int wubu_ir_init(wubu_irplaybook_t *p) {
    if (!p) return -1;
    p->state = WUBU_IR_IDLE;
    p->n_actions = 0;
    return 0;
}

wubu_ir_state_t wubu_ir_step(wubu_irplaybook_t *p, int step_ok) {
    if (!p) return WUBU_IR_IDLE;
    if (!step_ok) {
        /* fail-closed: a failed step halts the runbook (state unchanged, no
         * half-response). The operator must intervene. */
        return p->state;
    }
    switch (p->state) {
        case WUBU_IR_IDLE:
            p->state = WUBU_IR_KILLED;       /* step 1: kill */
            p->n_actions++;
            break;
        case WUBU_IR_KILLED:
            p->state = WUBU_IR_CONTAINED;    /* step 2: contain */
            p->n_actions++;
            break;
        case WUBU_IR_CONTAINED:
            p->state = WUBU_IR_ROLLED_BACK;  /* step 3: rollback */
            p->n_actions++;
            break;
        case WUBU_IR_ROLLED_BACK:
            /* already resolved: no further steps */
            break;
    }
    return p->state;
}

wubu_ir_state_t wubu_ir_state(const wubu_irplaybook_t *p) {
    return p ? p->state : WUBU_IR_IDLE;
}

int wubu_ir_resolved(const wubu_irplaybook_t *p) {
    return p ? (p->state == WUBU_IR_ROLLED_BACK ? 1 : 0) : 0;
}

int wubu_ir_reset(wubu_irplaybook_t *p) {
    if (!p) return -1;
    p->state = WUBU_IR_IDLE;
    p->n_actions = 0;
    return 0;
}
