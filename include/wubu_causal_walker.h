#ifndef WUBU_CAUSAL_WALKER_H
#define WUBU_CAUSAL_WALKER_H
/* wubu_anomaly.h companion — AGENT #311: causal walker localizes a divergence.
 *
 * Given a set of N cells with a causal dependency graph (a cell's output
 * feeds the next stage), and an observed divergence at a downstream cell,
 * the walker walks the causal chain BACKWARD to the earliest cell whose
 * own signal is anomalous — the "responsible cell" (root cause).
 *
 * Model: each cell has a value + an anomaly z-score (from the immune
 * system). The causal order is 0..N-1 (cell i feeds i+1). Given a
 * divergence reported at cell d, the walker scans forward from cell 0 to
 * d and returns the first (earliest) cell whose own signal is anomalous —
 * the root cause. If none, returns -1 (the divergence is external /
 * not attributable to any cell).
 *
 * C11, self-contained, fixed capacity. Covered by test_causal_walker with
 * measured assertions.
 */
#include <stddef.h>

#define WUBU_WALKER_MAX_CELLS 64

typedef struct {
    int     n;                          /* number of cells (<= MAX) */
    double  z[WUBU_WALKER_MAX_CELLS];   /* per-cell anomaly z-score */
    double  threshold;                  /* a cell is anomalous if z > this */
} wubu_walker_t;

/* Initialise with n cells and an anomaly threshold. Returns 0 ok / -1. */
int wubu_walker_init(wubu_walker_t *w, int n, double threshold);

/* Set a cell's anomaly z-score. Returns 0 ok / -1. */
int wubu_walker_set_z(wubu_walker_t *w, int cell, double z);

/* Given a divergence reported at cell d, walk backward to the earliest
 * anomalous cell <= d (the root cause). Returns the cell index, or -1
 * if no cell up to d is anomalous (unattributable). */
int wubu_walker_localize(const wubu_walker_t *w, int d);

/* Convenience: is a given cell anomalous (z > threshold)? */
int wubu_walker_anomalous(const wubu_walker_t *w, int cell);

#endif
