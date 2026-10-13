#ifndef WUBU_OOMQUAR_H
#define WUBU_OOMQUAR_H

/* wubu_oomquar.h -- KERNEL #14 memory quarantine on OOM.
 *
 * On an out-of-memory condition the kernel must NOT panic globally: the OOM is
 * scoped to the cgroup/service that exceeded its memory limit, and reclamation
 * is quarantined to that scope. This module models the scoped OOM handler:
 * per-scope memory accounting, an OOM killer that selects the largest memory
 * consumer within the over-limit scope, quarantines (frees) its pages, and
 * only escalates to a kernel panic if the scope still cannot be brought under
 * its limit after reclamation. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_OOM_MAX_SCOPES 32
#define WUBU_OOM_MAX_TASKS 128
#define WUBU_OOM_NAME_MAX 48

typedef struct {
    int id;
    char name[WUBU_OOM_NAME_MAX];
    int scope;               /* owning scope index, or -1 for kernel */
    uint64_t pages;          /* pages currently held */
    int quarantined;         /* 1 once selected + freed by the OOM killer */
    int pinned;              /* 1 if unreclaimable (kernel-resident) */
} wubu_oom_task_t;

typedef struct {
    char name[WUBU_OOM_NAME_MAX];
    uint64_t limit_pages;    /* memory limit for this scope */
    uint64_t used_pages;     /* sum of its tasks' pages */
    int oom_count;           /* times this scope hit OOM */
    int panic;               /* 1 if this scope forced a kernel panic */
} wubu_oom_scope_t;

typedef struct {
    wubu_oom_scope_t scopes[WUBU_OOM_MAX_SCOPES];
    int n_scopes;
    wubu_oom_task_t tasks[WUBU_OOM_MAX_TASKS];
    int n_tasks;
    int kernel_panic;        /* 1 if any scope escalated to a global panic */
    uint64_t total_quarantined; /* pages reclaimed across all OOM events */
} wubu_oomquar_t;

/* Initialise. Returns 0. */
int wubu_oom_init(wubu_oomquar_t *o);

/* Register a memory scope (cgroup/service) with a page limit. Returns 0/-1. */
int wubu_oom_add_scope(wubu_oomquar_t *o, const char *name, uint64_t limit_pages);

/* Find a scope index by name, or -1. */
int wubu_oom_find_scope(const wubu_oomquar_t *o, const char *name);

/* Add a task holding `pages` in `scope_name`. Returns task id, or -1. */
int wubu_oom_add_task(wubu_oomquar_t *o, const char *name, const char *scope_name,
                      uint64_t pages);

/* Handle an OOM in `scope_name`: if the scope is over its limit, select the
 * largest non-quarantined task in that scope, free its pages (quarantine), and
 * recompute. Repeat until under limit or no reclaimable task remains. If the
 * scope cannot be brought under limit, mark it (and the kernel) panicked.
 * Returns the number of pages quarantined, or -1 on bad args. A scope already
 * under limit returns 0 (no OOM). */
int64_t wubu_oom_handle(wubu_oomquar_t *o, const char *scope_name);

/* Is the scope currently over its limit? 1/0/-1. */
int wubu_oom_over_limit(const wubu_oomquar_t *o, const char *scope_name);

/* Did any OOM escalate to a kernel panic? 1/0. */
int wubu_oom_kernel_panicked(const wubu_oomquar_t *o);

#endif
