/*
 * wubu_credit.h -- Turn-level credit assignment (AH12).
 */
#ifndef WUBU_CREDIT_H
#define WUBU_CREDIT_H

double wubu_turn_credit(double prev_progress, double cur_progress,
                        double reward, double gamma);
int    wubu_credit_sign(double credit, double eps);
double wubu_credit_sum(const double *credits, int n);

#endif
