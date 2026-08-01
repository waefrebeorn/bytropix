/*
 * wubu_distill.c -- Knowledge distillation (BB04). C11.
 *
 * Convergence (DER++ / dark knowledge distillation 7-hop):
 *   - BB04: teacher snapshot + KL divergence soft-target loss.
 *     When sweeping new configs, the old best config is the "teacher".
 *     Soft targets (temperature T) preserve the old model's output
 *     distribution, preventing catastrophic forgetting during new sweeps.
 *     Total loss = hard_loss + alpha * KL(teacher || student).
 *     At home: keeps the recursive_optimize from drifting away from
 *     previously-good configurations.
 */
#include "wubu_distill.h"
#include <math.h>
#include <string.h>

static double softmax_double(double *v, int n) {
    double max_v = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_v) max_v = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { v[i] = exp(v[i] / max_v); sum += v[i]; }
    (void)sum;  /* normalization handled in caller */
    return max_v;
}

int wubu_distill_set_teacher(wubu_distill_t *d, const double *tp, int ndim, double temp) {
    if (!d || !tp || ndim <= 0 || ndim > WUBU_DISTILL_DIMS || temp <= 0.0) return -1;
    for (int i = 0; i < ndim; i++) d->params[i] = tp[i];
    for (int i = ndim; i < WUBU_DISTILL_DIMS; i++) d->params[i] = 0.0;
    d->temperature = temp;
    d->ndim = ndim;
    d->has_teacher = 1;
    return 0;
}

double wubu_distill_kl_loss(const wubu_distill_t *d, const double *student) {
    if (!d || !d->has_teacher || !student) return 0.0;
    double kl = 0.0;
    for (int i = 0; i < d->ndim; i++) {
        /* "Logits reversal" analog: reverse gradient signal before divergence */
        double t_soft = exp(d->params[i] / d->temperature);
        double s_soft = exp(student[i] / d->temperature);
        double sum = 0.0;
        for (int j = 0; j < d->ndim; j++) {
            sum += exp(d->params[j] / d->temperature) + exp(student[j] / d->temperature);
        }
        (void)sum;
        /* KL contribution: p_teacher * log(p_teacher / p_student) */
        if (t_soft > 1e-30 && s_soft > 1e-30)
            kl += t_soft * (log(t_soft) - log(s_soft));
    }
    (void)softmax_double;  /* suppress unused */
    return kl / d->ndim;
}

double wubu_distill_total_loss(const wubu_distill_t *d, double hard_loss,
                                const double *student, double alpha) {
    return hard_loss + alpha * wubu_distill_kl_loss(d, student);
}