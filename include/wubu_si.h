/*
 * wubu_si.h -- Synaptic Intelligence (BB06). C11.
 *
 * Zenke et al 2017: the per-parameter importance omega_i is the
 * PATH-INTEGRAL of the gradient along the training trajectory:
 *   omega_i += (theta_i(t) - theta_i(t-1)) * (-grad_i(t))
 * (the loss-gradient's contribution swept along the parameter's path).
 * The consolidation penalty is the EWC-style quadratic weighted by
 * omega, normalized by the total importance:
 *   L_si = lambda * sum_i( omega_i * (theta_i - anchor_i)^2 )
 *          / (sum_j omega_j + eps)
 */
#ifndef WUBU_SI_H
#define WUBU_SI_H

#define WUBU_SI_DIMS 15

typedef struct {
    double omega[WUBU_SI_DIMS];   /* per-parameter path-integral importance */
    double anchor[WUBU_SI_DIMS];  /* reference params (consolidation point) */
    double lambda;                /* regularization strength */
    double omega_sum;             /* sum_j omega_j (normalizer) */
    int    ndim;
    int    initialized;
} wubu_si_t;

/* Initialize with the current params as the anchor (first task). */
int wubu_si_init(wubu_si_t *s, const double *params, int ndim, double lambda);
/* Accumulate the path integral from a training step:
 * prev = the params BEFORE the step, curr = AFTER, grads = the loss
 * gradients used. omega_i += (curr_i - prev_i) * (-grads_i).
 * Returns the new omega_sum. */
double wubu_si_update(wubu_si_t *s, const double *prev, const double *curr,
                      const double *grads, int ndim);
/* The SI penalty at the current params (0 until the first update). */
double wubu_si_penalty(const wubu_si_t *s, const double *params);

#endif
