#ifndef WUBU_BOOTBUDGET_H
#define WUBU_BOOTBUDGET_H
#include <stdint.h>

/* wubu_bootbudget.h -- KERNEL #22 boot time budgeted + reported.
 *
 * Boot phases are timed against a budget: each phase (firmware, PCI probe,
 * memory init, driver load, kernel start, userspace) has a budget in ticks and
 * an actual measured duration. A phase that exceeds its budget is a BREACH
 * (flagged). The whole boot reports total time and whether any phase breached.
 * Deterministic, real work (no stubs). */

#define WUBU_BB_MAX 16

typedef struct {
    char name[24];
    uint64_t budget;     /* allowed ticks */
    uint64_t actual;     /* measured ticks */
} wubu_bb_phase_t;

typedef struct {
    wubu_bb_phase_t phases[WUBU_BB_MAX];
    int n;
} wubu_bootbudget_t;

/* Initialise. */
int wubu_bb_init(wubu_bootbudget_t *b);

/* Add a boot phase with its budget. Returns 0/-1. */
int wubu_bb_add(wubu_bootbudget_t *b, const char *name, uint64_t budget);

/* Record a phase's measured duration. Returns 1 if this phase BREACHED its
 * budget, 0 if within budget, -1 on error. */
int wubu_bb_record(wubu_bootbudget_t *b, const char *name, uint64_t actual);

/* Total measured boot time (sum of recorded phases). */
uint64_t wubu_bb_total(const wubu_bootbudget_t *b);

/* Total budget (sum of phase budgets). */
uint64_t wubu_bb_total_budget(const wubu_bootbudget_t *b);

/* Count of phases that breached their budget. */
int wubu_bb_breaches(const wubu_bootbudget_t *b);

/* Did the boot stay within its total budget? 1/0. */
int wubu_bb_within(const wubu_bootbudget_t *b);

#endif
