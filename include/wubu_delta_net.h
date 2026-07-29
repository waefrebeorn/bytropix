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

/* Chunkwise WY-form prefill (matches serial recurrence, O(d^2*C) not O(d^2*C*n)). */
void wubu_delta_net_chunk_prefill(const float *q, const float *k, const float *v,
                                  const float *beta, int n, int d, int chunk,
                                  float *S /* in/out d*d */);

/* Output gate: RMSNorm(S_out) * SiLU(gate_logits). */
void wubu_delta_net_apply_gate(const float *S_out, const float *gate_logits,
                               int n, int d, float *y);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DELTA_NET_H */
