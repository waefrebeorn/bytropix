/*
 * wubu_moeroute.c -- MoE capacity routing + load balancing (HH03). C11.
 *
 * Convergence (Switch Transformer / Expert Choice / capacity routing 7-hop):
 *   - HH03: top-k routing with capacity factor C (each expert ≤ C tokens).
 *     Overflow tokens (expert over capacity) are dropped → residual. Auxiliary
 *     load-balancing loss (importance + aux) prevents expert collapse. At home:
 *     the MoE layer (wubu_moe) routes tokens to experts with capacity caps +
 *     load-balancing → compute-balanced MoE inference (no idle experts, no
 *     overflow thrash) → faster MoE decode, serving the throughput mandate.
 */
#include "wubu_moeroute.h"
#include <string.h>
#include <math.h>

int wubu_moeroute_init(wubu_moeroute_t *mr, int n_experts, int top_k, int capacity) {
    if (!mr || n_experts <= 0 || n_experts > WUBU_MOEROUTE_MAX_EXPERTS) return -1;
    if (top_k <= 0 || top_k > n_experts) return -1;
    if (capacity <= 0) return -1;
    memset(mr, 0, sizeof(*mr));
    mr->n_experts = n_experts;
    mr->top_k = top_k;
    mr->capacity = capacity;
    return 0;
}

int wubu_moeroute_step(wubu_moeroute_t *mr, int n_tokens) {
    if (!mr || n_tokens <= 0 || n_tokens > WUBU_MOEROUTE_MAX_TOKENS) return -1;
    mr->total_routed = 0;
    mr->dropped = 0;
    memset(mr->load, 0, sizeof(mr->load));
    for (int t = 0; t < n_tokens; t++) {
        /* top-k over router_logits[t] */
        int chosen[WUBU_MOEROUTE_MAX_EXPERTS];
        int n_chosen = 0;
        /* greedy top-k */
        for (int k = 0; k < mr->top_k; k++) {
            int best = -1; float best_v = -1e30f;
            for (int e = 0; e < mr->n_experts; e++) {
                int used = 0;
                for (int c = 0; c < n_chosen; c++) if (chosen[c] == e) used = 1;
                if (used) continue;
                if (mr->router_logits[t][e] > best_v) { best_v = mr->router_logits[t][e]; best = e; }
            }
            if (best < 0) break;
            /* capacity check */
            if (mr->load[best] < mr->capacity) {
                chosen[n_chosen++] = best;
                mr->load[best]++;
                mr->total_routed++;
            }
            /* if over capacity, skip (dropped) */
        }
        if (n_chosen < mr->top_k) mr->dropped += (mr->top_k - n_chosen);
    }
    return mr->total_routed;
}

float wubu_moeroute_aux_loss(const wubu_moeroute_t *mr) {
    if (!mr || mr->n_experts == 0) return 0.0f;
    /* aux loss = Σ_e f_e · P_e where f_e = fraction routed, P_e = mean router prob.
       Simplified: variance of load across experts (lower = more balanced). */
    float mean = (float)mr->total_routed / (float)mr->n_experts;
    float var = 0.0f;
    for (int e = 0; e < mr->n_experts; e++) {
        float d = (float)mr->load[e] - mean;
        var += d * d;
    }
    var /= mr->n_experts;
    return var;  /* higher variance = worse balance → loss to minimize */
}
