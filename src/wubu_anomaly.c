/* wubu_anomaly.c — AGENT #310: hive diagnostic immune system.
 * C11, self-contained. Typed measurement cells + rolling-window z-score
 * anomaly detection.
 */
#include "wubu_anomaly.h"
#include <math.h>

int wubu_anomaly_init(wubu_anomaly_t *a, double z_threshold) {
    if (!a) return -1;
    for (int i = 0; i < WUBU_MEAS_KIND_COUNT; i++) {
        a->cell[i].kind = (wubu_meas_kind_t)i;
        a->cell[i].n = 0;
        a->cell[i].head = 0;
        a->cell[i].mean = 0.0;
        a->cell[i].sd = 0.0;
        a->cell[i].anomalies = 0;
        for (int j = 0; j < WUBU_ANOMALY_WINDOW; j++)
            a->cell[i].win[j] = 0.0;
    }
    a->z_threshold = z_threshold > 0 ? z_threshold : WUBU_ANOMALY_Z;
    return 0;
}

int wubu_anomaly_record(wubu_anomaly_t *a, wubu_meas_kind_t kind, double value) {
    if (!a || kind < 0 || kind >= WUBU_MEAS_KIND_COUNT) return -1;
    wubu_meas_cell_t *c = &a->cell[kind];
    c->win[c->head] = value;
    c->head = (c->head + 1) % WUBU_ANOMALY_WINDOW;
    if (c->n < WUBU_ANOMALY_WINDOW) c->n++;
    /* recompute mean/sd over the live window */
    double sum = 0.0;
    for (int i = 0; i < c->n; i++) sum += c->win[i];
    c->mean = sum / (double)c->n;
    double var = 0.0;
    for (int i = 0; i < c->n; i++) {
        double d = c->win[i] - c->mean;
        var += d * d;
    }
    c->sd = sqrt(var / (double)c->n);
    return 0;
}

int wubu_anomaly_score(const wubu_anomaly_t *a, wubu_meas_kind_t kind, double *z) {
    if (!a || kind < 0 || kind >= WUBU_MEAS_KIND_COUNT) return -1;
    const wubu_meas_cell_t *c = &a->cell[kind];
    if (c->n < 2 || c->sd <= 0.0) return -1;   /* need variance */
    /* the most recent value is at (head-1+W)%W */
    int last = (c->head - 1 + WUBU_ANOMALY_WINDOW) % WUBU_ANOMALY_WINDOW;
    double v = c->win[last];
    double zz = fabs(v - c->mean) / c->sd;
    if (z) *z = zz;
    return zz > a->z_threshold ? 1 : 0;
}

int wubu_anomaly_check(wubu_anomaly_t *a, wubu_meas_kind_t kind, double value, double *z) {
    if (wubu_anomaly_record(a, kind, value) != 0) return -1;
    int r = wubu_anomaly_score(a, kind, z);
    if (r == 1) a->cell[kind].anomalies++;
    return r < 0 ? 0 : r;   /* insufficient samples -> not anomalous */
}

int wubu_anomaly_fired(const wubu_anomaly_t *a) {
    if (!a) return 0;
    int total = 0;
    for (int i = 0; i < WUBU_MEAS_KIND_COUNT; i++)
        total += a->cell[i].anomalies;
    return total;
}

int wubu_anomaly_count(const wubu_anomaly_t *a, wubu_meas_kind_t kind) {
    if (!a || kind < 0 || kind >= WUBU_MEAS_KIND_COUNT) return -1;
    return a->cell[kind].n;
}
