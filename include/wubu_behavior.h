#ifndef WUBU_BEHAVIOR_H
#define WUBU_BEHAVIOR_H
#include <stddef.h>
#include "wubu_anomaly.h"
#include "wubu_safekern.h"

/* wubu_behavior.h -- AGI-EDR runtime behavior classifier (AGIEDR #295).
 *
 * Ties the anomaly immune system (wubu_anomaly: rolling z-score per metric)
 * to the containment ladder (wubu_safekern: NONE/WARN/THROTTLE/SUSPEND/STOP).
 * The classifier consumes a stream of agent behavior metrics, folds the worst
 * per-metric z-score into a single severity in [0,1], maps severity to a
 * containment level, and issues a FAIL-CLOSED halt: if the classifier itself
 * is uninitialised or a metric is NaN, it defaults to the most severe level
 * (STOP) rather than silently allowing the agent to continue.
 *
 * Design discipline: kernel-owned stop, deterministic, no allocation in the
 * hot path, real work (no stubs). */

/* Number of consecutive anomalous samples that escalate one containment level. */
#define WUBU_BEHAVIOR_ESCALATE 2

typedef struct {
    wubu_anomaly_t  anom;       /* rolling z-score detector per metric */
    double          severity;   /* last-computed severity in [0,1] */
    int             level;      /* last-computed containment level */
    int             halt;       /* 1 = agent must stop (fail-closed) */
    int             streak;     /* consecutive anomalous observations */
    int             steps;      /* total observations classified */
} wubu_behavior_t;

/* Initialise the classifier. z_threshold is the anomaly |z| fire point
 * (e.g. 3.0). Returns 0 on success, -1 on NULL. */
int wubu_behavior_init(wubu_behavior_t *b, double z_threshold);

/* Observe one agent behavior sample for a metric kind. Records it in the
 * anomaly detector, recomputes severity/level/halt. Returns the containment
 * level after this observation. Fail-closed: NULL -> WUBU_CONT_STOP. */
int wubu_behavior_observe(wubu_behavior_t *b, wubu_meas_kind_t kind, double value);

/* Current fail-closed halt verdict: 1 = stop the agent now, 0 = may continue. */
int wubu_behavior_halt(const wubu_behavior_t *b);

/* Current containment level (WUBU_CONT_*). */
int wubu_behavior_level(const wubu_behavior_t *b);

/* Current severity in [0,1]. */
double wubu_behavior_severity(const wubu_behavior_t *b);

/* Reset the streak/severity (e.g. after an operator clears a WARN). */
void wubu_behavior_reset(wubu_behavior_t *b);

/* Tripwire kill-switch bridge (AGIEDR #294): when the classifier halts
 * (SUSPEND/STOP), ARM the kernel-owned stop_flag on the safekern. The stop
 * is one-way: once armed it cannot be cleared by the agent (wubu_safekern
 * exposes no setter/clearer to the reasoner). Returns 1 if the stop is now
 * armed, 0 otherwise. Fail-closed: NULL behavior halts and arms. */
int wubu_behavior_arm_killswitch(const wubu_behavior_t *b, wubu_safekern_t *k);

#endif
