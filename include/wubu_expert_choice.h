/*
 * wubu_expert_choice.h -- Fine-grained MoE expert-choice routing (doc E05).
 *
 * Expert Choice routing: each expert picks top-k tokens (vs standard
 * top-k routing where each token picks top-k experts).
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_EXPERT_CHOICE_H
#define WUBU_EXPERT_CHOICE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Standard top-k routing: each token picks top-k experts. */
void wubu_topk_route(const float *scores, int n_tokens, int n_experts, int k,
                     int *out_assignments, float *out_weights);

/* Expert Choice routing: each expert picks top-k tokens. */
void wubu_expert_choice_route(const float *scores, int n_tokens, int n_experts, int k,
                                int *out_assignments, float *out_weights);

/* Compute load balance (coefficient of variation). Lower = more balanced. */
float wubu_route_load_balance(const int *assignments, int n_experts, int k, int n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EXPERT_CHOICE_H */
