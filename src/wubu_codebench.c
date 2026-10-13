/* wubu_codebench.c -- PATHS #327 verifiable code generation benchmark.
 * See wubu_codebench.h. */
#include "wubu_codebench.h"
#include <string.h>

int wubu_cb_init(wubu_codebench_t *b) {
    if (!b) return -1;
    memset(b->tasks, 0, sizeof b->tasks);
    b->n_tasks = 0;
    b->last_passed = 0;
    b->last_total_cases = 0;
    return 0;
}

int wubu_cb_add_task(wubu_codebench_t *b, const char *name, int opcode,
                     const wubu_cb_case_t *cases, int n_cases) {
    if (!b || !name || !cases || b->n_tasks >= WUBU_CB_MAX_TASKS) return -1;
    if (n_cases <= 0 || n_cases > WUBU_CB_MAX_CASES) return -1;
    if (strlen(name) >= 32) return -1;
    wubu_cb_task_t *t = &b->tasks[b->n_tasks];
    strncpy(t->name, name, 31); t->name[31] = '\0';
    t->opcode = opcode;
    t->n_cases = n_cases;
    memcpy(t->cases, cases, sizeof(wubu_cb_case_t) * n_cases);
    b->n_tasks++;
    return 0;
}

long long wubu_cb_exec(int opcode, long long input) {
    switch (opcode) {
        case 0: return input;
        case 1: return input + 1;
        case 2: return input * 2;
        case 3: return input * input;
        case 4: return -input;
        default: return 0;
    }
}

int wubu_cb_run(wubu_codebench_t *b, int *out_cases) {
    if (!b) return -1;
    int passed = 0, total_cases = 0;
    for (int i = 0; i < b->n_tasks; i++) {
        wubu_cb_task_t *t = &b->tasks[i];
        int all_ok = 1;
        for (int j = 0; j < t->n_cases; j++) {
            total_cases++;
            long long got = wubu_cb_exec(t->opcode, t->cases[j].input);
            if (got != t->cases[j].expected) { all_ok = 0; }
        }
        if (all_ok) passed++;
    }
    b->last_passed = passed;
    b->last_total_cases = total_cases;
    if (out_cases) *out_cases = total_cases;
    return passed;
}

double wubu_cb_pass_rate(const wubu_codebench_t *b) {
    if (!b || b->n_tasks == 0) return 0.0;
    return (double)b->last_passed / (double)b->n_tasks;
}
