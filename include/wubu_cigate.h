#ifndef WUBU_CIGATE_H
#define WUBU_CIGATE_H

/* wubu_cigate.h -- STACK #368 three-repo integration CI gate.
 *
 * A CI gate that runs the real test gates of all three WuBu repos
 * (wubunos = compiler, wubuwizard = brain, wubuos = kernel) and reports an
 * aggregate green/red verdict. Each leg invokes the repo's actual gate
 * command (gauntlet_runner --core for the compiler; per-module test binaries
 * for brain/kernel) via fork+exec, captures the exit status, and rolls up.
 * The integration gate is GREEN only if every leg is green. Deterministic,
 * real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_CI_CMD_MAX 1024
#define WUBU_CI_OUT_MAX 8192

typedef enum {
    WUBU_CI_GREEN = 0,   /* all legs passed */
    WUBU_CI_RED = 1      /* at least one leg failed / errored */
} wubu_ci_verdict_t;

typedef struct {
    char name[32];           /* "compiler" / "brain" / "kernel" */
    char cmd[WUBU_CI_CMD_MAX]; /* the gate command that was run */
    int ran;                 /* 1 if the command was executed */
    int exit_code;           /* child exit code (0 = pass) */
    int passed;              /* 1 if exit_code == 0 */
    char output[WUBU_CI_OUT_MAX]; /* captured combined stdout/stderr (tail) */
} wubu_ci_leg_t;

typedef struct {
    wubu_ci_leg_t legs[3];   /* [0]=compiler [1]=brain [2]=kernel */
    int n_legs;
    int n_passed;
    int n_failed;
    wubu_ci_verdict_t verdict;
} wubu_ci_report_t;

/* Initialise a report. Returns 0. */
int wubu_ci_init(wubu_ci_report_t *r);

/* Run one gate leg: fork+exec `cmd` (via /bin/sh -c), capture combined
 * stdout/stderr (tail-bounded), record exit code + pass/fail. `name` labels
 * the leg. Returns 0 on success (regardless of the leg's own pass/fail),
 * -1 on args error. */
int wubu_ci_run_leg(wubu_ci_leg_t *leg, const char *name, const char *cmd);

/* Roll up a report from its legs: count passed/failed, set verdict. */
int wubu_ci_finalize(wubu_ci_report_t *r);

/* Human-readable one-line summary of a report. Writes into `buf`. */
void wubu_ci_summary(const wubu_ci_report_t *r, char *buf, size_t cap);

#endif
