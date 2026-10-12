#ifndef WUBU_ANOMALY_H
#define WUBU_ANOMALY_H
/* wubu_anomaly.h — AGENT #310: hive = diagnostic immune system.
 *
 * The hive's immune layer: a set of TYPED measurement cells (LOSS,
 * GRAD_NORM, ACT_NORM, ...) each holding a rolling window of recent
 * values. A new sample is scored against its cell's window with a
 * z-score; |z| > threshold flags an anomaly (the immune system firing
 * on a divergence it has not seen).
 *
 * C11, self-contained, fixed capacity. The forward (record + score +
 * anomaly flag) is covered by test_anomaly with measured assertions.
 */
#include <stddef.h>

/* The measurement cell types the immune system tracks. */
typedef enum {
    WUBU_MEAS_LOSS = 0,
    WUBU_MEAS_GRAD_NORM,
    WUBU_MEAS_ACT_NORM,
    WUBU_MEAS_LOGIT_ENTROPY,
    WUBU_MEAS_KIND_COUNT
} wubu_meas_kind_t;

#define WUBU_ANOMALY_WINDOW 32     /* rolling window per cell */
#define WUBU_ANOMALY_Z 3.0         /* |z| above this fires */

typedef struct {
    wubu_meas_kind_t kind;
    double  win[WUBU_ANOMALY_WINDOW];
    int     n;            /* samples in window (<= WUBU_ANOMALY_WINDOW) */
    int     head;         /* ring write index */
    double  mean, sd;     /* last-computed stats */
    int     anomalies;    /* count of anomalous samples */
} wubu_meas_cell_t;

typedef struct {
    wubu_meas_cell_t cell[WUBU_MEAS_KIND_COUNT];
    double  z_threshold;
} wubu_anomaly_t;

/* Initialise (no samples). Returns 0 ok / -1 bad. */
int wubu_anomaly_init(wubu_anomaly_t *a, double z_threshold);

/* Record a sample for a cell kind. Returns 0 ok / -1 bad. */
int wubu_anomaly_record(wubu_anomaly_t *a, wubu_meas_kind_t kind, double value);

/* Score the most recent sample of a cell against its window: writes the
 * z-score to *z and returns 1 if anomalous (|z| > threshold), 0 if not,
 * -1 on bad args / insufficient samples (< 2). */
int wubu_anomaly_score(const wubu_anomaly_t *a, wubu_meas_kind_t kind, double *z);

/* Convenience: record + score in one step (the immune check). Returns
 * 1 if THIS sample is anomalous, 0 if not / too few samples, -1 bad. */
int wubu_anomaly_check(wubu_anomaly_t *a, wubu_meas_kind_t kind, double value, double *z);

/* Total anomalies fired across all cells since init. */
int wubu_anomaly_fired(const wubu_anomaly_t *a);

/* Number of samples in a cell's window. */
int wubu_anomaly_count(const wubu_anomaly_t *a, wubu_meas_kind_t kind);

#endif
