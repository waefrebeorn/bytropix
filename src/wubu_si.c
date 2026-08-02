/*
 * wubu_si.c -- Synaptic Intelligence (BB06). C11.
 */
#include "wubu_si.h"
#include <string.h>
#include <math.h>

int wubu_si_init(wubu_si_t *s, const double *params, int ndim, double lambda)
{
    if (!s || !params || ndim <= 0 || ndim > WUBU_SI_DIMS || lambda < 0)
        return -1;
    memset(s->omega, 0, sizeof(s->omega));
    memcpy(s->anchor, params, (size_t)ndim * sizeof(double));
    s->lambda = lambda;
    s->omega_sum = 0;
    s->ndim = ndim;
    s->initialized = 1;
    return 0;
}

double wubu_si_update(wubu_si_t *s, const double *prev, const double *curr,
                      const double *grads, int ndim)
{
    if (!s || !s->initialized || !prev || !curr || !grads)
        return s ? s->omega_sum : 0;
    if (ndim > s->ndim) ndim = s->ndim;
    for (int i = 0; i < ndim; i++) {
        /* the path-integral term: omega += delta_w * (-grad) */
        s->omega[i] += (curr[i] - prev[i]) * (-grads[i]);
    }
    s->omega_sum = 0;
    for (int i = 0; i < s->ndim; i++) s->omega_sum += s->omega[i];
    if (s->omega_sum < 0) s->omega_sum = 0;
    return s->omega_sum;
}

double wubu_si_penalty(const wubu_si_t *s, const double *params)
{
    if (!s || !s->initialized || !params) return 0;
    if (s->omega_sum <= 0) return 0;
    double num = 0;
    for (int i = 0; i < s->ndim; i++) {
        double d = params[i] - s->anchor[i];
        num += s->omega[i] * d * d;
    }
    return s->lambda * num / (s->omega_sum + 1e-12);
}
