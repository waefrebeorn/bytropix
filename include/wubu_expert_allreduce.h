/*
 * wubu_expert_allreduce.h — Wide-expert all-reduce reference (doc E06, CPU core).
 *
 * In wide-expert MoE (>=8 experts sharded across hosts/GPUs), each rank computes
 * its local expert outputs and a final all-reduce sums the partials. The
 * reduction math (ring or tree all-reduce) is pure CPU; only the transport
 * (NCCL/IB) is HW. We implement the reduction op + a simulated multi-rank
 * all-reduce so the numerics are testable on one machine. Ties to C05.
 */
#ifndef WUBU_EXPERT_ALLREDUCE_H
#define WUBU_EXPERT_ALLREDUCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sum-reduce `nranks` partial vectors [len] each into out[...] (the all-reduce
 * result). out may alias partials[0]. */
void wubu_allreduce_sum(const float *const *partials, int nranks, int len, float *out);

/* Ring all-reduce (Inception-style): processes `len` in chunks of `chunk`;
 * verifies the result equals the plain sum. Returns max abs diff vs sum. */
float wubu_ring_allreduce_check(const float *const *partials, int nranks,
                                 int len, int chunk, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EXPERT_ALLREDUCE_H */
