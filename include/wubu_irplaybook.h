#ifndef WUBU_IRPLAYBOOK_H
#define WUBU_IRPLAYBOOK_H

/* wubu_irplaybook.h -- AGIEDR #307 agent incident-response playbook.
 *
 * An incident-response playbook: when the EDR detects a threat, the agent runs
 * a fixed runbook in ORDER -- (1) KILL the affected process, (2) CONTAIN the
 * blast radius (isolate), (3) ROLLBACK to the last-known-good state. Each step
 * must complete before the next; a step that fails halts the runbook
 * (fail-closed) so the agent never half-responds. Tracks the response state.
 * Deterministic, real work (no stubs). */

typedef enum {
    WUBU_IR_IDLE=0,       /* no incident */
    WUBU_IR_KILLED,       /* step 1 done: process killed */
    WUBU_IR_CONTAINED,    /* step 2 done: blast radius contained */
    WUBU_IR_ROLLED_BACK,  /* step 3 done: state restored (resolved) */
} wubu_ir_state_t;

typedef struct {
    wubu_ir_state_t state;
    int n_actions;        /* runbook steps executed */
} wubu_irplaybook_t;

/* Initialise (idle). */
int wubu_ir_init(wubu_irplaybook_t *p);

/* Execute the next runbook step. 'step_ok' = whether the step succeeds (the
 * environment can make a step fail). Returns the new state, or WUBU_IR_IDLE
 * with a halt if a step fails (fail-closed: runbook halts on first failure). */
wubu_ir_state_t wubu_ir_step(wubu_irplaybook_t *p, int step_ok);

/* Current state. */
wubu_ir_state_t wubu_ir_state(const wubu_irplaybook_t *p);

/* Is the incident fully resolved (rolled back)? 1/0. */
int wubu_ir_resolved(const wubu_irplaybook_t *p);

/* Reset to idle for the next incident. */
int wubu_ir_reset(wubu_irplaybook_t *p);

#endif
