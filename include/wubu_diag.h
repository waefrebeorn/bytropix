/*
 * wubu_diag.h -- THE HIVE DIAGNOSTIC SYSTEM (research/056, INDEX AN08).
 *
 * The user's directive (2026-08-04): "we need to design the superior AGI
 * diagnostic system using our hive structure. The hive will save us."
 *
 * THE HIVE IS THE DIAGNOSTIC SYSTEM. Every measurement the AGI takes
 * (loss, grads, route, oracle score, system vitals) is a CELL in the same
 * hive tissue as the model's own cells. Diagnosis = walking that tissue;
 * mutation = growing/shrinking that tissue; the 5+1 recovery = replaying
 * that tissue. One structure, every level, memory-bounded by construction
 * -- the hive cannot bloat (the lesson of the 103-checkpoint / 15 GiB
 * archive is baked into the structure).
 *
 *   LOSS     train / held-out loss + ema        (the trainer, every step)
 *   GRAD     per-cell grad norms                (the REAL backprop)
 *   ENTROPY  route entropy per layer            (moe2/hashrouter)
 *   ROUTE    which cells fired per token        (the forward -- the route
 *                                                trace = self-explanation)
 *   UTIL     per-cell utilization               (the router's counter)
 *   BI       block importance                   (wubu_bi)
 *   ORACLE   RLHF / NVIDIA score_draft          (the live loop)
 *   DATA     corpus stream health               (the corpus mixer)
 *   SYS      disk / RAM / GPU / SSD free        (the foundry's vitals)
 *   MUT      the mutation ledger: grow/shrink + fitness delta (amoeba)
 *
 * The ring discipline: insert = wubu_hive_insert; when the hive is at
 * capacity the OLDEST cell (min step) recycles automatically. The trace
 * keeps the last ~capacity measurements -- NEVER the whole history.
 *
 * C11, self-contained, no third-party. Wraps wubu_hive (the tissue).
 */
#ifndef WUBU_DIAG_H
#define WUBU_DIAG_H

#include <stdint.h>
#include <stddef.h>
#include "wubu_hive.h"

typedef enum {
    WUBU_DIAG_LOSS = 0,
    WUBU_DIAG_GRAD,
    WUBU_DIAG_ENTROPY,
    WUBU_DIAG_ROUTE,
    WUBU_DIAG_UTIL,
    WUBU_DIAG_BI,
    WUBU_DIAG_ORACLE,
    WUBU_DIAG_DATA,
    WUBU_DIAG_SYS,
    WUBU_DIAG_MUT,
    WUBU_DIAG_NKINDS
} wubu_diag_kind;

/* one typed measurement: { kind, step, cell, value, meta } */
typedef struct {
    wubu_diag_kind kind;
    int64_t step;      /* the training step (or logical event order) */
    int cell;          /* which cell/expert/component (-1 = global) */
    float value;       /* the measurement */
    float meta;        /* kind-specific: ema, op code, fitness delta, ... */
} wubu_diag_cell;

/* per-kind aggregate state over the live window (incremental) */
typedef struct {
    int64_t n;         /* measurements seen (live) */
    double  sum;       /* sum of values */
    double  sumsq;     /* sum of squares (for std) */
} wubu_diag_agg;

typedef struct wubu_diag wubu_diag_t;

/* default ring capacity (measurements kept per live window) */
#define WUBU_DIAG_DEFAULT_CAPACITY 4096
/* z-score anomaly threshold (|z| > 2.5 = out of family) */
#define WUBU_DIAG_Z_THRESH 2.5f
/* absolute grad floor (the DA bug: relative-only misses the all-dead
 * colony -- a dead colony must classify SHRINK, not stasis) */
#define WUBU_DIAG_GRAD_FLOOR 1e-4f

/* init the diagnostic over an EXISTING hive (the tissue). kinds = bitmask
 * of enabled wubu_diag_kind (0 = all). Returns NULL on alloc failure. */
wubu_diag_t *wubu_diag_init(wubu_hive_t *hive, unsigned kinds);

/* ring capacity: how many measurements the trace keeps. Default
 * WUBU_DIAG_DEFAULT_CAPACITY. Set BEFORE recording. */
void wubu_diag_set_capacity(wubu_diag_t *d, size_t capacity);

/* record one measurement: malloc a cell + hive_insert. When the hive is
 * at capacity, the OLDEST cell (min step) recycles first (the ring).
 * Returns 0 on success. */
int wubu_diag_record(wubu_diag_t *d, wubu_diag_kind kind, int cell,
                     float value, float meta);

/* z-score of `value` against the kind's live-window distribution.
 * (x - mean)/std; 0 when std == 0 or no data. */
float wubu_diag_zscore(const wubu_diag_t *d, wubu_diag_kind kind,
                       float value);

/* the immune system over the window (per GRAD cell):
 *   grow   = a cell whose latest grad z-score is out of family (overworked)
 *   shrink = a cell below the absolute floor for its WHOLE live window
 *            (dead -- the DA bug guard)
 * *grow / *shrink receive counts (0 if NULL). Returns 0 on success. */
int wubu_diag_classify(wubu_diag_t *d, float *grow, float *shrink);

/* ---- REAL-GRAD BRIDGE (milestone 2) ---- */
#include "wubu_train.h"
int wubu_diag_record_grads(wubu_diag_t *d, const wubu_train_t *tr);

/* THE CAUSAL WALKER: on a fitness drop (held-out loss past loss_tol),
 * find the EARLIEST out-of-family measurement that precedes it -- the
 * root cause candidate. report[] gets a human-readable diagnosis, or the
 * honest "no out-of-family measurement found; fitness drop unexplained".
 * Returns 1 when a cause was found, 0 when none, -1 on error. */
int wubu_diag_walk(wubu_diag_t *d, int64_t drop_step, char *report,
                   size_t cap);

/* dump the diagnostic trace as JSON (aggregates + live cells). */
int wubu_diag_snapshot(wubu_diag_t *d, const char *json_path);

/* live measurement count (== hive live). */
size_t wubu_diag_live(const wubu_diag_t *d);

/* free all cells (hive_clear) + the diag struct. The hive itself stays
 * owned by the caller. */
void wubu_diag_free(wubu_diag_t *d);

#endif /* WUBU_DIAG_H */
