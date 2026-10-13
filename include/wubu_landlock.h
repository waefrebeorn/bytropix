#ifndef WUBU_LANDLOCK_H
#define WUBU_LANDLOCK_H
#include <stddef.h>
#include <stdint.h>

/* wubu_landlock.h -- OMARCHY #276 Landlock unprivileged filesystem sandbox.
 *
 * Models a Landlock LSM ruleset: a set of path-scoped access rules that grant
 * a confined process specific rights (read/write/execute/remove/...) under
 * specific directory subtrees, with DEFAULT-DENY for everything else. This
 * is the unprivileged sandbox model (no root needed) -- the policy core; the
 * actual landlock_create_ruleset() syscall is the kernel's job. Fail-closed:
 * an unmatched path/right, a NULL ruleset, or an uninitialised ruleset
 * denies. Deterministic, no allocation, real work (no stubs). */

/* Access rights (bitmask, mirroring Landlock's fs_access set conceptually). */
typedef enum {
    WUBU_LL_EXECUTE   = 1u<<0,
    WUBU_LL_WRITE     = 1u<<1,
    WUBU_LL_READ      = 1u<<2,
    WUBU_LL_READ_DIR  = 1u<<3,
    WUBU_LL_REMOVE    = 1u<<4,
    WUBU_LL_MAKE_DIR  = 1u<<5,
    WUBU_LL_MAKE_REG  = 1u<<6,
} wubu_ll_right_t;

#define WUBU_LL_MAX_RULES 64
#define WUBU_LL_PATHLEN   256

typedef struct {
    char     path[WUBU_LL_PATHLEN];  /* subtree root the rule applies under */
    unsigned rights;                 /* granted rights bitmask */
} wubu_ll_rule_t;

typedef struct {
    wubu_ll_rule_t rules[WUBU_LL_MAX_RULES];
    int n_rules;
    unsigned handled;   /* rights the ruleset governs (others always denied) */
} wubu_landlock_t;

/* Initialise an empty default-deny ruleset governing all rights. */
int wubu_landlock_init(wubu_landlock_t *l);
/* Add a rule: grant `rights` under `path` (a subtree root). Returns 0/-1. */
int wubu_landlock_allow(wubu_landlock_t *l, const char *path, unsigned rights);
/* Check: is `right` granted for `path`? A path is governed by a rule if it is
 * within that rule's subtree. Fail-closed 1/0. */
int wubu_landlock_check(const wubu_landlock_t *l, const char *path, wubu_ll_right_t right);

#endif
