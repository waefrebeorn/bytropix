#ifndef WUBU_ROADMAP_H
#define WUBU_ROADMAP_H
#include <stdint.h>

/* wubu_roadmap.h -- PATHS #329 AGI roadmap milestone tracker.
 *
 * An AGI-roadmap milestone tracker: milestones with a target quarter and a
 * completion status. The tracker computes roadmap progress (fraction done),
 * overdue milestones (target passed, not done), and an overall risk signal
 * (on-track / at-risk / behind) from the ratio of overdue to remaining.
 * Deterministic, real work (no stubs). */

#define WUBU_RM_MAX 64
#define WUBU_RM_NAMELEN 48

typedef enum {
    WUBU_RM_NOT_STARTED=0,
    WUBU_RM_IN_PROGRESS,
    WUBU_RM_DONE,
} wubu_rm_status_t;

typedef struct {
    char name[WUBU_RM_NAMELEN];
    int target_q;          /* target quarter (e.g. 2026Q3 = 20263) */
    wubu_rm_status_t status;
} wubu_rm_milestone_t;

typedef struct {
    wubu_rm_milestone_t ms[WUBU_RM_MAX];
    int n;
    int current_q;         /* current quarter */
} wubu_roadmap_t;

/* Initialise with the current quarter. */
int wubu_rm_init(wubu_roadmap_t *r, int current_q);

/* Add a milestone. Returns 0/-1. */
int wubu_rm_add(wubu_roadmap_t *r, const char *name, int target_q, wubu_rm_status_t status);

/* Mark a milestone done. Returns 0/-1. */
int wubu_rm_complete(wubu_roadmap_t *r, const char *name);

/* Fraction of milestones done (in [0,1]). */
double wubu_rm_progress(const wubu_roadmap_t *r);

/* Count of overdue milestones (target_q < current_q and not done). */
int wubu_rm_overdue(const wubu_roadmap_t *r);

/* Roadmap health: 0 = on-track, 1 = at-risk (some overdue), 2 = behind
 * (overdue > 25% of milestones). */
int wubu_rm_health(const wubu_roadmap_t *r);

#endif
