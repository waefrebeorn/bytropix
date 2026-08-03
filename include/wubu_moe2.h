/*
 * wubu_moe2.h -- the mixed-agents router for the WuBu model (phase 2).
 *
 * The blueprint: the seed's FFN becomes fine-grained experts -- many
 * small specialized "agents", few active per token (DeepSeekMoE
 * pattern). This module is sized for the 35M seed: 8 experts, 2
 * active, plus a shared expert always on. The router is the committee
 * that decides which agents speak.
 *
 * The wizard's wubu_moe (256 experts, GPU kernels) is the production
 * path; this is the seed-sized drop-in that plugs into wubu's
 * blocks so the 35M model can grow the agent structure now.
 */
#ifndef WUBU_MOE2_H
#define WUBU_MOE2_H

#include <stdint.h>

#define MOE2_N_EXPERTS   8
#define MOE2_N_ACTIVE    2
#define MOE2_D_MODEL     448
#define MOE2_D_FF        256

/* the mixed-agents weights for one block */
typedef struct {
    /* router: [MOE2_D_MODEL, MOE2_N_EXPERTS] */
    float *router_w;
    /* experts: gate/up [D_MODEL, D_FF] and down [D_FF, D_MODEL] each */
    float *exp_gate[MOE2_N_EXPERTS];
    float *exp_up[MOE2_N_EXPERTS];
    float *exp_down[MOE2_N_EXPERTS];
    /* the shared expert (always active) */
    float *shared_gate, *shared_up, *shared_down;
    /* the shared-expert gate scalar */
    float *shared_gate_w;   /* [D_MODEL] */
} wubu_moe2_t;

/* router: scores [N_EXPERTS] = softmax(x @ router_w) ; picks top-k
 * active indices + weights. Returns the count active (k). */
int wubu_moe2_route(const wubu_moe2_t *moe, const float *x,
                    int active_idx[MOE2_N_ACTIVE],
                    float active_w[MOE2_N_ACTIVE]);

/* forward one token through the mixed agents:
 * out[o] = shared(x) + Σ_active w_e · expert_e(x)
 * x: [D_MODEL], out: [D_MODEL]. Returns 0 on success. */
int wubu_moe2_forward(const wubu_moe2_t *moe, const float *x, float *out);

/* init weights (small random) -- the trainer initializes these. */
int wubu_moe2_init(wubu_moe2_t *moe, uint32_t seed);
void wubu_moe2_free(wubu_moe2_t *moe);

#endif
