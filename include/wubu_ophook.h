#ifndef WUBU_OPHOOK_H
#define WUBU_OPHOOK_H
#include <stdint.h>

/* wubu_ophook.h -- AGENT #151 operator hook applies/restarts/directive.
 *
 * The operator hook: the channel through which a human operator (or the Body)
 * applies a promoted mutation, restarts the affected component, or issues a
 * directive (a config/policy change). It is a small command state machine:
 * APPLY (commit a mutation), RESTART (bounce a component), DIRECTIVE (set a
 * policy). Each command is validated against the current state (fail-closed:
 * e.g. cannot restart a component that isn't applied) and recorded with a
 * sequence number + applied flag. Deterministic, no allocation, real work. */

typedef enum {
    WUBU_OP_APPLY=0,    /* apply a promoted mutation */
    WUBU_OP_RESTART,    /* restart the affected component */
    WUBU_OP_DIRECTIVE,  /* issue a policy/config directive */
} wubu_op_cmd_t;

#define WUBU_OPHOOK_MAX 128
#define WUBU_OPHOOK_ARGLEN 64

typedef struct {
    wubu_op_cmd_t cmd;
    char arg[WUBU_OPHOOK_ARGLEN];   /* mutation id / component / directive */
    int  applied;                   /* 1 = executed, 0 = rejected */
    uint64_t seq;                   /* monotonic sequence number */
} wubu_op_entry_t;

typedef struct {
    wubu_op_entry_t log[WUBU_OPHOOK_MAX];
    int  n;
    uint64_t next_seq;
    int  n_applied;
    int  n_rejected;
} wubu_ophook_t;

/* Initialise the operator hook. */
int wubu_ophook_init(wubu_ophook_t *h);

/* Issue an operator command. APPLY/DIRECTIVE always succeed; RESTART requires
 * the named component to have been APPLIED first (fail-closed otherwise).
 * Returns the entry index or -1. */
int wubu_ophook_issue(wubu_ophook_t *h, wubu_op_cmd_t cmd, const char *arg);

/* Was the given component applied (so it can be restarted)? 1/0. */
int wubu_ophook_applied(const wubu_ophook_t *h, const char *component);

/* Stats. */
int wubu_ophook_applied_count(const wubu_ophook_t *h);
int wubu_ophook_rejected_count(const wubu_ophook_t *h);

#endif
