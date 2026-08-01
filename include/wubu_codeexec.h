/*
 * wubu_codeexec.h -- Code exec verifier → feeds loopguard (AX07).
 */
#ifndef WUBU_CODEEXEC_H
#define WUBU_CODEEXEC_H

typedef struct {
    int last_rc;
    int last_oom;
    int last_latency_us;
    int last_verified;
} wubu_codeexec_t;

int wubu_codeexec_init(wubu_codeexec_t *ce);
int wubu_codeexec_verify(const wubu_codeexec_t *ce,
                                     const char *source,
                                     int latency_budget_us);
int wubu_codeexec_run_regression(const char *source,
                                               int *out_rc,
                                               int *out_oom,
                                               long *out_latency_us);

#endif