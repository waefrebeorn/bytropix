/*
 * wubu_acq.c -- Acquisition functions (FF02). C11.
 *
 * Convergence (EI / UCB / PI 7-hop):
 *   - EI(x) = (μ-f*)Φ((μ-f*)/σ) + σ·φ((μ-f*)/σ)  [closed form]
 *   - UCB(x) = μ + κ·σ
 *   - PI(x) = Φ((μ-f*)/σ)
 *   At home: these score candidate configs; BO maximizes them to pick the next
 *   sweep to run. Balances exploit (high μ) with explore (high σ).
 */
#include "wubu_acq.h"
#include <math.h>

/* Standard normal CDF (Abramowitz-Stegun approximation). */
static double norm_cdf(double z) {
    return 0.5 * (1.0 + erf(z / sqrt(2.0)));
}

double wubu_acq_value(const wubu_acq_t *acq, double mean, double std) {
    if (!acq || std < 0) return 0.0;
    if (std < 1e-9) {
        /* Degenerate: only mean matters */
        switch (acq->type) {
            case WUBU_ACQ_UCB: return mean + acq->kappa * std;
            case WUBU_ACQ_EI:  return (mean > acq->f_star) ? mean - acq->f_star : 0.0;
            case WUBU_ACQ_PI:  return (mean > acq->f_star) ? 1.0 : 0.0;
        }
        return mean;
    }
    double z = (mean - acq->f_star) / std;
    switch (acq->type) {
        case WUBU_ACQ_EI: {
            /* EI with exploration slack xi */
            double zz = (mean - acq->f_star - acq->xi) / std;
            double pdf = exp(-0.5 * zz * zz) / sqrt(2.0 * M_PI);
            return (mean - acq->f_star - acq->xi) * norm_cdf(zz) + std * pdf;
        }
        case WUBU_ACQ_UCB:
            return mean + acq->kappa * std;
        case WUBU_ACQ_PI:
            return norm_cdf(z);
    }
    return mean;
}
