#ifndef WUBU_KDA_H
#define WUBU_KDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kimi Delta Attention: S_t = D_t(I - k k^T)S + k v^T, D=diag(decay), decay per-channel. */
void wubu_kda_recurrence(const float *q, const float *k, const float *v,
                         const float *decay, int n, int d, float *S);
void wubu_kda_output(const float *q, const float *S, int n, int d, float *y);
int  wubu_kda_state_bounded(const float *S, int d, float *l2);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KDA_H */
