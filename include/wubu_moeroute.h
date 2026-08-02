/*
 * wubu_moeroute.h -- MoE capacity routing + load balancing (HH03).
 */
#ifndef WUBU_MOEROUTE_H
#define WUBU_MOEROUTE_H

#define WUBU_MOEROUTE_MAX_EXPERTS 64
#define WUBU_MOEROUTE_MAX_TOKENS 256

typedef struct {
    int n_experts;
    int top_k;
    int capacity;            /* max tokens per expert (capacity factor × batch) */
    /* Load stats */
    int load[WUBU_MOEROUTE_MAX_EXPERTS];   /* tokens routed this step */
    int total_routed;
    int dropped;             /* overflow tokens (residual) */
    /* Affinity: router logits per (token, expert) for top-k selection. */
    float router_logits[WUBU_MOEROUTE_MAX_TOKENS][WUBU_MOEROUTE_MAX_EXPERTS];
    /* Load-balancing loss accumulator */
    float aux_loss;          /* aux + importance loss */
} wubu_moeroute_t;

/* Init router with capacity C per expert (C = capacity_factor × batch). */
int  wubu_moeroute_init(wubu_moeroute_t *mr, int n_experts, int top_k, int capacity);
/* Route one batch: assign each token to top-k experts under capacity.
   Returns number of tokens successfully routed (non-dropped). */
int  wubu_moeroute_step(wubu_moeroute_t *mr, int n_tokens);
/* Compute load-balancing aux loss (encourages uniform expert usage). */
float wubu_moeroute_aux_loss(const wubu_moeroute_t *mr);

#endif