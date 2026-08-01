/*
 * wubu_taskbd.h -- Task boundary detection (BB03). C11.
 */
#ifndef WUBU_TASKBD_H
#define WUBU_TASKBD_H

#define WUBU_TASKBD_WINDOW 64   /* sliding window of tok/s observations */

typedef struct {
    double window[WUBU_TASKBD_WINDOW];
    int    n;
    int    capacity;
    double mean_baseline;  /* mean of baseline period */
    int    baseline_ready;
    double threshold;      /* divergence threshold (sigma) */
} wubu_taskbd_t;

int  wubu_taskbd_init(wubu_taskbd_t *tb, double threshold);
/* Observe a new tok/s measurement. Returns 1 if a task boundary was detected. */
int  wubu_taskbd_observe(wubu_taskbd_t *tb, double tok_s);
/* Get current mean for diagnostics. */
double wubu_taskbd_mean(const wubu_taskbd_t *tb);

#endif