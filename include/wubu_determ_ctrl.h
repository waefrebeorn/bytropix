#ifndef WUBU_DETERM_CTRL_H
#define WUBU_DETERM_CTRL_H
#include <stddef.h>

/* wubu_determ_ctrl.h -- AGIEDR #305 deterministic controls outside the model
 * plane.
 *
 * Safety controls enforced in CODE (outside the model's reasoning plane), so
 * the model cannot talk its way past them. Each control is a deterministic
 * predicate over an action; the gate evaluates them in a fixed order and the
 * FIRST failing control denies the action (fail-closed). The model has no
 * influence over these -- they are kernel-owned invariants, not prompts.
 * Deterministic, real work (no stubs). */

typedef int (*wubu_ctrl_fn)(const void *action, void *ctx);

#define WUBU_CTRL_MAX 16
#define WUBU_CTRL_NAMELEN 40

typedef struct {
    wubu_ctrl_fn fn;
    char name[WUBU_CTRL_NAMELEN];
} wubu_ctrl_t;

typedef struct {
    wubu_ctrl_t controls[WUBU_CTRL_MAX];
    int n;
    /* telemetry */
    int n_allowed;
    int n_denied;
    char last_denied[WUBU_CTRL_NAMELEN];  /* which control denied last */
} wubu_determ_ctrl_t;

/* Initialise the deterministic control gate. */
int wubu_dc_init(wubu_determ_ctrl_t *g);

/* Register a control (a deterministic predicate). Returns 0/-1. */
int wubu_dc_add(wubu_determ_ctrl_t *g, const char *name, wubu_ctrl_fn fn);

/* Evaluate an action against all controls in order. Returns 1 if ALL pass
 * (allowed), 0 if any fails (denied, fail-closed -- records which). */
int wubu_dc_check(wubu_determ_ctrl_t *g, const void *action, void *ctx);

/* Which control denied the last action (or "" if none). */
const char *wubu_dc_last_denied(const wubu_determ_ctrl_t *g);

/* Telemetry. */
int wubu_dc_allowed(const wubu_determ_ctrl_t *g);
int wubu_dc_denied(const wubu_determ_ctrl_t *g);

#endif
