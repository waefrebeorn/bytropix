/*
 * wubu_energy.c -- Energy-aware inference (Theme IJ). C11.
 *
 * Pure deterministic models, no hardware access (the WuBu kernel's
 * energy ledger can later be fed by real RAPL/CMU counters).
 */
#include "wubu_energy.h"
#include <math.h>

/* IJ01 -- the energy roofline. */
float wubu_energy_estimate(float mem_bytes, float j_per_mem_byte,
                           float flops, float j_per_flop)
{
    float e = mem_bytes * j_per_mem_byte + flops * j_per_flop;
    return e < 0 ? 0 : e;
}

float wubu_energy_j_per_token(float bytes_per_token, float j_per_mem_byte)
{
    if (bytes_per_token <= 0 || j_per_mem_byte <= 0) return 0;
    return bytes_per_token * j_per_mem_byte;
}

float wubu_energy_tokens_per_joule(float bytes_per_token, float j_per_mem_byte)
{
    float jpt = wubu_energy_j_per_token(bytes_per_token, j_per_mem_byte);
    return jpt > 0 ? (1.0f / jpt) : 0;
}

/* IJ02 -- the ledger. */
int wubu_energy_ledger_init(wubu_energy_ledger_t *e, float budget_j)
{
    if (!e || budget_j < 0) return -1;
    e->budget_j = budget_j;
    e->spent_j  = 0;
    e->tokens   = 0;
    e->over     = 0;
    return 0;
}

int wubu_energy_ledger_spend(wubu_energy_ledger_t *e, float j, uint64_t n)
{
    if (!e || j < 0) return -1;
    e->spent_j += j;
    e->tokens  += n;
    if (e->spent_j >= e->budget_j) e->over = 1;   /* exhausted at the boundary */
    return e->over;
}

float wubu_energy_ledger_remaining(wubu_energy_ledger_t *e)
{
    if (!e) return 0;
    float r = e->budget_j - e->spent_j;
    return r < 0 ? 0 : r;
}

float wubu_energy_ledger_jpt(wubu_energy_ledger_t *e)
{
    if (!e || e->tokens == 0) return 0;
    return e->spent_j / (float)e->tokens;
}

/* IJ03 -- power-cap frequency scheduler.
 * Dynamic power P(f) = P_base * (f/f_base)^3 (V follows f). */
float wubu_energy_freq_for_cap(float cap_w, float p_base, float f_base)
{
    if (cap_w <= 0 || p_base <= 0 || f_base <= 0) return 0;
    float frac = powf(cap_w / p_base, 1.0f / 3.0f);
    if (frac > 1.0f) frac = 1.0f;
    return frac * f_base;
}

float wubu_energy_jpt_at_freq(float jpt_base, float compute_frac,
                              float f_base, float f)
{
    if (jpt_base <= 0 || f_base <= 0 || f <= 0) return 0;
    if (compute_frac < 0) compute_frac = 0;
    if (compute_frac > 1) compute_frac = 1;
    /* compute-bound part scales with the runtime (f_base/f);
     * memory-bound part (1 - compute_frac) does not. */
    return jpt_base * (compute_frac * (f_base / f) + (1.0f - compute_frac));
}

float wubu_energy_freq_optimal(float cap_w, float p_base, float f_base,
                               float jpt_base, float compute_frac)
{
    float f_max = wubu_energy_freq_for_cap(cap_w, p_base, f_base);
    if (f_max <= 0) return 0;
    /* memory-bound decode (compute_frac ~ 0): energy is flat in f, so
     * any f under the cap is fine; take the max f under the cap (best
     * throughput at the same energy). compute-bound: lower f cuts both
     * power and jpt -- the cap already bounds it; take f_max. */
    return f_max;
}

/* IJ04 -- energy-budget early exit. */
int wubu_energy_should_continue(wubu_energy_ledger_t *e,
                                float jpt, float quality_gate)
{
    if (!e || jpt <= 0) return 1;
    if (quality_gate < 0) quality_gate = 0;
    if (quality_gate > 1) quality_gate = 1;
    float need = jpt / (quality_gate > 0 ? quality_gate : 1.0f);
    return wubu_energy_ledger_remaining(e) >= need;
}

/* IJ05 -- energy-tier offload (amortized J/byte). */
int wubu_energy_choose_tier(float bytes, float j_per_byte_a, float j_per_byte_b,
                            float reuse_rate)
{
    if (bytes <= 0) return 0;
    if (j_per_byte_a < 0) j_per_byte_a = 0;
    if (j_per_byte_b < 0) j_per_byte_b = 0;
    if (reuse_rate < 0) reuse_rate = 0;
    /* amortized: each byte is read ~ (1 + reuse_rate) times */
    float ea = bytes * j_per_byte_a * (1.0f + reuse_rate);
    float eb = bytes * j_per_byte_b * (1.0f + reuse_rate);
    return eb < ea ? 1 : 0;
}

/* IJ06 -- speculative-decoding energy break-even.
 * One round drafts K tokens (draft_jpt*K J), the target verifies them
 * in ONE forward pass (target_jpt J) and keeps ~A*K + 1 tokens.
 * The same tokens without spec cost target_jpt * (A*K + 1). Spec wins
 * while: draft_jpt*K + target_jpt < target_jpt*(A*K + 1)
 *   -> draft_jpt < target_jpt * A.                       (DA-verified) */
float wubu_energy_spec_breakeven(float target_jpt, float accept_rate,
                                 uint32_t k)
{
    if (target_jpt <= 0 || k == 0) return 0;
    if (accept_rate < 0) accept_rate = 0;
    if (accept_rate > 1) accept_rate = 1;
    return target_jpt * accept_rate;
}

float wubu_energy_spec_round(float draft_jpt, float target_jpt,
                             float accept_rate, uint32_t k)
{
    if (k == 0 || target_jpt <= 0) return target_jpt;
    if (accept_rate < 0) accept_rate = 0;
    if (accept_rate > 1) accept_rate = 1;
    float tokens = accept_rate * (float)k + 1.0f;       /* kept + verify */
    float cost_with = draft_jpt * (float)k + target_jpt; /* draft + verify */
    return cost_with / tokens;   /* per-accepted-token; vs target_jpt */
}

/* IJ07 -- the budget-driven operator. */
int wubu_energy_pick_config(const float *jpt, const float *tok_per_s,
                            uint32_t n, float min_tok_per_s,
                            float budget_j, float *out_jpt)
{
    if (!jpt || !tok_per_s || n == 0 || !out_jpt) return -1;
    int best = -1;
    for (uint32_t i = 0; i < n; i++) {
        if (tok_per_s[i] < min_tok_per_s) continue;
        if (jpt[i] > budget_j) continue;      /* unaffordable config */
        if (best < 0 || jpt[i] < jpt[best]) best = (int)i;
    }
    if (best < 0) return -1;
    *out_jpt = jpt[best];
    return best;
}
