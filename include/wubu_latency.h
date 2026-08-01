/*
 * wubu_latency.h -- AGI-OS latency-class scheduler (AF05-AF07).
 */
#ifndef WUBU_LATENCY_H
#define WUBU_LATENCY_H

typedef enum { WUBU_LC_HRT = 0, WUBU_LC_SRT = 1, WUBU_LC_DT = 2 } wubu_latclass_t;

typedef struct {
    int   id;
    long  deadline_ms;
    long  exec_ms;
} wubu_task_t;

typedef struct {
    long   wcet_ms;
    double mean_ms;
    double jitter_ms;
} wubu_wcet_t;

typedef struct {
    long   ttft_ms;     /* time to first token */
    long   turn_ms;     /* full-turn latency */
    double jitter_ms;
    double tok_per_sec;
} wubu_slo_meas_t;

int  wubu_edf_order(wubu_task_t *t, int n);            /* AF05 EDF sort */
void wubu_wcet_account(const long *samples, int n, wubu_wcet_t *out); /* AF06 */
int  wubu_deadline_miss(const wubu_wcet_t *w, long budget_ms);        /* AF06 */
int  wubu_slo_check(wubu_latclass_t cls, const wubu_slo_meas_t *m);    /* AF07 */

#endif
