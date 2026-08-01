/*
 * wubu_distill.h -- Knowledge distillation (BB04). C11.
 */
#ifndef WUBU_DISTILL_H
#define WUBU_DISTILL_H

#define WUBU_DISTILL_DIMS 15

typedef struct {
    double params[WUBU_DISTILL_DIMS];  /* teacher (snapshot) params */
    double temperature; /* softening temperature T */
    int    ndim;
    int    has_teacher;
} wubu_distill_t;

int  wubu_distill_set_teacher(wubu_distill_t *d, const double *teacher_params, int ndim, double temperature);
/* KL-div soft-target loss: sum over dims of: p*log(p/q) where p=soft(teacher), q=soft(student). */
double wubu_distill_kl_loss(const wubu_distill_t *d, const double *student_params);
/* Total loss = hard_loss (raw tok_s) + alpha * kl_loss */
double wubu_distill_total_loss(const wubu_distill_t *d, double hard_loss,
                                const double *student_params, double alpha);

#endif