/*
 * wubu_energy.h -- Energy-aware inference (Theme IJ: power-budgeted
 * decode). C11, opaque-style API, no third-party deps.
 *
 * Convergence (7-hop: HPC power-capping / DVFS / energy roofline /
 * edge-LLM energy):
 *   - IJ01 Energy model: the energy roofline -- E ~ mem_bytes * J/byte
 *         + flops * J/flop; decode is memory-bound, so E/token tracks
 *         the bytes moved (mirrors the perf roofline spine).
 *   - IJ02 Energy-per-token ledger: cumulative J accounting with a
 *         hard budget (arXiv 2603.20224 "Advocating Energy-per-Token").
 *   - IJ03 Power-cap frequency scheduler: DVFS P ~ C*V^2*f; memory-bound
 *         decode means LOWER frequency raises tok/J (CCGrid 2026
 *         characterization) -- pick the frequency under a power cap.
 *   - IJ04 Energy-budget early exit: stop decode when the remaining
 *         budget cannot afford the next token at the required quality.
 *   - IJ05 Energy-tier KV offload: choose the storage tier by the
 *         amortized J/byte (DRAM vs NVMe), not just capacity.
 *   - IJ06 Speculative-decoding energy break-even: the drafter's J/token
 *         vs the accepted-token savings (draft cheaper than the
 *         rejected-token energy it replaces).
 *   - IJ07 Integration: the budget-driven operator -- pick the
 *         energy-optimal config on the J/token frontier.
 */
#ifndef WUBU_ENERGY_H
#define WUBU_ENERGY_H

#include <stddef.h>
#include <stdint.h>

/* IJ01 -- energy roofline model.
 * E = mem_bytes * j_per_mem_byte + flops * j_per_flop. */
float wubu_energy_estimate(float mem_bytes, float j_per_mem_byte,
                           float flops, float j_per_flop);
/* J per token (mem-bound decode): bytes_per_token * j_per_mem_byte. */
float wubu_energy_j_per_token(float bytes_per_token, float j_per_mem_byte);
/* tokens per joule (the reciprocal; the edge metric). */
float wubu_energy_tokens_per_joule(float bytes_per_token, float j_per_mem_byte);

/* IJ02 -- the energy ledger with a hard budget (J). */
typedef struct wubu_energy_ledger {
    float budget_j;      /* total energy allowance */
    float spent_j;       /* cumulative spend */
    uint64_t tokens;     /* tokens produced */
    int    over;         /* 1 once the budget is exhausted */
} wubu_energy_ledger_t;

int   wubu_energy_ledger_init(wubu_energy_ledger_t *e, float budget_j);
/* Spend j for n tokens. Returns 0 if the budget remains, 1 if over. */
int   wubu_energy_ledger_spend(wubu_energy_ledger_t *e, float j, uint64_t n);
float wubu_energy_ledger_remaining(wubu_energy_ledger_t *e);
float wubu_energy_ledger_jpt(wubu_energy_ledger_t *e);   /* avg J/token */

/* IJ03 -- power-cap frequency scheduler (DVFS P ~ C*V^2*f).
 * Dynamic power at freq f: P(f) = P_base * (f/f_base)^3 for the
 * voltage-follows-frequency regime (V ~ f). Returns the highest
 * frequency (fraction of f_base, <= 1) that stays under cap_w. */
float wubu_energy_freq_for_cap(float cap_w, float p_base, float f_base);
/* Memory-bound decode: J/token is ~const across f (DRAM energy is
 * frequency-independent); time/token scales as 1/f only when
 * compute-bound. The model: jpt_at_f = jpt_base * (compute_frac +
 * (1 - compute_frac) * f_base/f) -- compute-bound part scales with the
 * time, memory-bound part does not. */
float wubu_energy_jpt_at_freq(float jpt_base, float compute_frac,
                              float f_base, float f);
/* Energy-optimal frequency under a cap: the f that minimizes
 * J/token while staying under cap_w (the CCGrid finding: lower f =
 * higher tok/J for memory-bound decode, bounded by the cap). */
float wubu_energy_freq_optimal(float cap_w, float p_base, float f_base,
                               float jpt_base, float compute_frac);

/* IJ04 -- energy-budget early exit. Decode may stop when the remaining
 * budget cannot afford the next token at the minimum quality gate.
 * quality_gate in [0,1]: 1 = never exit early, 0 = exit as soon as the
 * budget is below one token. Returns 1 = keep decoding, 0 = stop. */
int wubu_energy_should_continue(wubu_energy_ledger_t *e,
                                float jpt, float quality_gate);

/* IJ05 -- energy-tier KV offload. Choose the storage tier with the
 * lower AMORTIZED energy: read_energy = bytes * j_per_byte * (1 +
 * reuse_rate * copies). Returns 0 for tier A, 1 for tier B. */
int wubu_energy_choose_tier(float bytes, float j_per_byte_a, float j_per_byte_b,
                            float reuse_rate);

/* IJ06 -- speculative-decoding energy break-even. Drafting K tokens
 * costs draft_jpt*K J; the target verifies them in ONE forward pass
 * (target_jpt J) and keeps ~A*K+1 tokens. Spec is energy-neutral while
 * draft_jpt < target_jpt * A (the drafter must be cheaper than the
 * accepted-token energy it replaces -- DA-verified model). Returns the
 * max drafter J/token that keeps spec energy-neutral (0 = never). */
float wubu_energy_spec_breakeven(float target_jpt, float accept_rate,
                                 uint32_t k);
/* Energy per accepted token of one spec round (draft + verify +
 * acceptance); the caller compares against target_jpt (no-spec). */
float wubu_energy_spec_round(float draft_jpt, float target_jpt,
                             float accept_rate, uint32_t k);

/* IJ07 -- the budget-driven operator: pick the config on the J/token
 * frontier under a budget. Configs are (jpt, tok_per_s) pairs; the
 * operator selects the lowest-jpt config that clears min_tok_per_s. */
int wubu_energy_pick_config(const float *jpt, const float *tok_per_s,
                            uint32_t n, float min_tok_per_s,
                            float budget_j, float *out_jpt);

#endif
