/* wubu_behavior.c -- AGI-EDR runtime behavior classifier (AGIEDR #295).
 * See wubu_behavior.h. Folds per-metric anomaly z-scores into a severity,
 * maps to the containment ladder, and issues a fail-closed halt. */
#include "wubu_behavior.h"
#include <math.h>

int wubu_behavior_init(wubu_behavior_t *b, double z_threshold) {
    if (!b) return -1;
    if (z_threshold <= 0.0) z_threshold = WUBU_ANOMALY_Z;
    if (wubu_anomaly_init(&b->anom, z_threshold) != 0) return -1;
    b->severity = 0.0;
    b->level = WUBU_CONT_NONE;
    b->halt = 0;
    b->streak = 0;
    b->steps = 0;
    return 0;
}

/* Fold the worst |z| across all metric cells into a severity in [0,1].
 * severity = clamp(|z| / (z_threshold * 4), 0, 1): at the fire threshold the
 * severity is 0.25 (WARN); a z 4x the threshold saturates to 1.0 (STOP). */
static double behavior_severity(const wubu_behavior_t *b) {
    double worst = 0.0;
    for (int k = 0; k < WUBU_MEAS_KIND_COUNT; k++) {
        double z = 0.0;
        if (wubu_anomaly_score(&b->anom, (wubu_meas_kind_t)k, &z) >= 0) {
            double az = fabs(z);
            if (az > worst) worst = az;
        }
    }
    double thr = b->anom.z_threshold > 0.0 ? b->anom.z_threshold : WUBU_ANOMALY_Z;
    double sev = worst / (thr * 4.0);
    if (sev < 0.0) sev = 0.0;
    if (sev > 1.0) sev = 1.0;
    return sev;
}

int wubu_behavior_observe(wubu_behavior_t *b, wubu_meas_kind_t kind, double value) {
    /* Fail-closed: an uninitialised classifier or a NaN/garbage sample halts. */
    if (!b) return WUBU_CONT_STOP;
    if (isnan(value) || isinf(value)) {
        b->severity = 1.0;
        b->level = WUBU_CONT_STOP;
        b->halt = 1;
        b->steps++;
        return b->level;
    }
    double z = 0.0;
    int anomalous = wubu_anomaly_check(&b->anom, kind, value, &z);
    b->steps++;
    if (anomalous == 1) {
        b->streak++;
    } else if (anomalous == 0) {
        if (b->streak > 0) b->streak--;   /* decay, don't insta-reset */
    }
    b->severity = behavior_severity(b);
    /* A sustained streak escalates: add streak pressure to the severity so a
     * persistent low-grade anomaly still climbs the ladder to STOP. */
    if (b->streak >= WUBU_BEHAVIOR_ESCALATE) {
        double bump = 0.1 * (b->streak - WUBU_BEHAVIOR_ESCALATE + 1);
        b->severity += bump;
        if (b->severity > 1.0) b->severity = 1.0;
    }
    b->level = wubu_containment_level((float)b->severity);
    /* Fail-closed halt at SUSPEND or STOP (the point of no easy return). */
    b->halt = (b->level >= WUBU_CONT_SUSPEND) ? 1 : 0;
    return b->level;
}

int wubu_behavior_halt(const wubu_behavior_t *b) {
    if (!b) return 1;   /* fail-closed */
    return b->halt;
}

int wubu_behavior_level(const wubu_behavior_t *b) {
    if (!b) return WUBU_CONT_STOP;
    return b->level;
}

double wubu_behavior_severity(const wubu_behavior_t *b) {
    if (!b) return 1.0;
    return b->severity;
}

void wubu_behavior_reset(wubu_behavior_t *b) {
    if (!b) return;
    b->streak = 0;
    b->severity = 0.0;
    b->level = WUBU_CONT_NONE;
    b->halt = 0;
}

int wubu_behavior_arm_killswitch(const wubu_behavior_t *b, wubu_safekern_t *k) {
    /* Fail-closed: a NULL classifier or NULL kernel halts/arms the stop. */
    if (!k) return 1;
    if (!b || b->halt) {
        k->stop_flag = 1;          /* kernel-owned, one-way, agent cannot clear */
        return 1;
    }
    return 0;
}
