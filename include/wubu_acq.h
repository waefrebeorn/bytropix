/*
 * wubu_acq.h -- Acquisition functions for Bayesian Optimization (FF02).
 */
#ifndef WUBU_ACQ_H
#define WUBU_ACQ_H

#define WUBU_ACQ_EI  0   /* Expected Improvement */
#define WUBU_ACQ_UCB 1   /* Upper Confidence Bound */
#define WUBU_ACQ_PI  2   /* Probability of Improvement */

typedef struct {
    int   type;       /* WUBU_ACQ_* */
    double f_star;    /* incumbent best observed value */
    double kappa;     /* UCB exploration weight */
    double xi;        /* EI exploration slack */
} wubu_acq_t;

/* Compute acquisition value given GP predictive (mean, std). */
double wubu_acq_value(const wubu_acq_t *acq, double mean, double std);

#endif