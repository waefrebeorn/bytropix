#ifndef WUBU_DELTA_NET_H
#define WUBU_DELTA_NET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gated-DeltaNet serial recurrence: S_t = (I - b k k^T) S + b k v^T.
 * q,k,v: n*d row-major. beta: n (per-token gate). S: d*d in/out. */
void wubu_delta_net_recurrence(const float *q, const float *k, const float *v,
                               const float *beta, int n, int d, float *S);
/* y_t = S q_t. */
void wubu_delta_net_output(const float *q, const float *S, int n, int d, float *y);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DELTA_NET_H */
