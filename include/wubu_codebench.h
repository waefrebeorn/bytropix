#ifndef WUBU_CODEBENCH_H
#define WUBU_CODEBENCH_H

/* wubu_codebench.h -- PATHS #327 verifiable code generation benchmark.
 *
 * A verifiable code-generation benchmark: a suite of coding tasks, each with a
 * set of test cases (input -> expected output). Generated code is executed
 * against the test cases; a task PASSES only if ALL its cases match (verifiable,
 * not vibes). The harness scores pass rate and per-task results. This is the
 * "verifiable" part -- correctness is checked mechanically, not judged.
 * Deterministic, real work (no stubs). */

#define WUBU_CB_MAX_TASKS 32
#define WUBU_CB_MAX_CASES 16
#define WUBU_CB_CODELEN 256

typedef struct {
    long long input;
    long long expected;
} wubu_cb_case_t;

typedef struct {
    char name[32];
    wubu_cb_case_t cases[WUBU_CB_MAX_CASES];
    int n_cases;
    /* the "generated code" as an op-code program (a tiny verifiable IR):
     * 0=return input, 1=return input+1, 2=return input*2, 3=return input*input,
     * 4=return -input */
    int opcode;
} wubu_cb_task_t;

typedef struct {
    wubu_cb_task_t tasks[WUBU_CB_MAX_TASKS];
    int n_tasks;
    int last_passed;
    int last_total_cases;
} wubu_codebench_t;

/* Initialise. */
int wubu_cb_init(wubu_codebench_t *b);

/* Add a task: name, opcode (the generated program), and its test cases.
 * Returns 0/-1. */
int wubu_cb_add_task(wubu_codebench_t *b, const char *name, int opcode,
                     const wubu_cb_case_t *cases, int n_cases);

/* Execute a program opcode on an input. */
long long wubu_cb_exec(int opcode, long long input);

/* Run all tasks; returns the number of tasks that PASS (all cases match).
 * Fills out_cases with the total cases checked. */
int wubu_cb_run(wubu_codebench_t *b, int *out_cases);

/* Pass rate over tasks (in [0,1]). */
double wubu_cb_pass_rate(const wubu_codebench_t *b);

#endif
