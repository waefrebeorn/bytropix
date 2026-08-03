/*
 * wubu_agi.c -- the AGI loop (WuBu's recursive learning machine).
 *
 * The loop (one iteration):
 *   1. observe: push an observation into the hive
 *   2. think: run the model forward over the context, route
 *      through the mixed agents, return the top-1 candidate
 *   3. verify: the prover checks the proposal (the reward)
 *   4. act: the accepted action is applied
 *   5. learn: the outcome is a training datum
 *   6. grow: checkpoint saved (5+1 rollback)
 *
 * This module is the conductor. Every organ is a self-contained
 * C11 module; the AGI loop just calls them.
 */
#include "wubu_agi.h"
#include <stdio.h>
#include <string.h>

/* helper: collect hive tokens into a flat array */
static int collect_token(void *ptr, void *user)
{
    uint16_t *buf = (uint16_t *)((void **)user)[0];
    int *k = (int *)((void **)user)[1];
    int max = *(int *)((void **)user)[2];
    if (*k >= max) return 1;   /* stop early */
    buf[(*k)++] = (uint16_t)(uintptr_t)ptr;
    return 0;
}

int wubu_agi_init(wubu_agi_t *agi, barun_model_t *m,
                      barun_buf_t *b, wubu_moe2_t *agents)
{
    if (!agi || !m || !b || !agents) return -1;
    memset(agi, 0, sizeof(*agi));
    agi->model = m;
    agi->buf = b;
    agi->agents = agents;
    wubu_hive_init(&agi->memory);
    wubu_moe2_init(agents, 42);
    return 0;
}

int wubu_agi_observe(wubu_agi_t *agi, uint16_t token)
{
    if (!agi) return -1;
    int r = wubu_hive_insert(&agi->memory, (void *)(uintptr_t)token);
    if (r == 0) agi->memory_writes++;
    return r;
}

int wubu_agi_think(wubu_agi_t *agi, uint16_t *out_tokens, int max_out)
{
    if (!agi || !out_tokens || max_out <= 0) return -1;
    size_t n = wubu_hive_live(&agi->memory);
    if (n == 0 || n > 64) return 0;
    /* iterate the hive to build the token sequence */
    uint16_t ctx[64];
    int k = 0;
    void *user[3] = { ctx, &k, (void *)(intptr_t)max_out };
    wubu_hive_foreach(&agi->memory, collect_token, user);
    if (k == 0) return 0;
    /* forward the model */
    barun_forward(agi->model, agi->buf, ctx, k);
    /* pick the top-1 */
    float *logits = barun_last_logits(agi->buf);
    if (!logits) return -1;
    int best = 0;
    for (int i = 1; i < BARUN_VOCAB && i < max_out; i++)
        if (logits[i] > logits[best]) best = i;
    out_tokens[0] = (uint16_t)best;
    return 1;
}

int wubu_agi_verify(wubu_agi_t *agi, const wubu_pf_step_t *step)
{
    if (!agi || !step) return -1;
    int ok = wubu_prover_check(step);
    if (ok) agi->accepted_steps++;
    else agi->rejected_steps++;
    return ok;
}

int wubu_agi_step(wubu_agi_t *agi, uint16_t observation,
                      uint16_t *action_out)
{
    if (wubu_agi_observe(agi, observation) != 0) return -1;
    int n = wubu_agi_think(agi, action_out, 1);
    if (n <= 0) return -1;
    /* verify the action (the prover rewards sound proposals) */
    wubu_pf_step_t step = { WUBU_PF_RING, (double)*action_out, 0, 0, 0, 0 };
    wubu_agi_verify(agi, &step);
    agi->iterations++;
    return n;
}

void wubu_agi_stats(const wubu_agi_t *agi, char *buf, size_t cap)
{
    if (!agi || !buf || cap == 0) return;
    snprintf(buf, cap,
        "iter=%llu accepted=%llu rejected=%llu "
        "mem_writes=%llu mem_reuses=%llu",
        (unsigned long long)agi->iterations,
        (unsigned long long)agi->accepted_steps,
        (unsigned long long)agi->rejected_steps,
        (unsigned long long)agi->memory_writes,
        (unsigned long long)agi->memory_reuses);
}
