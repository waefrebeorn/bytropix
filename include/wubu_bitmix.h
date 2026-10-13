#ifndef WUBU_BITMIX_H
#define WUBU_BITMIX_H
#include <stddef.h>

/* wubu_bitmix.h -- BRAIN #80 mixed per-role bit ladders, never uniform.
 *
 * Assigns a DIFFERENT bit-width to each weight role so the quantization is
 * never uniform: attention/embedding (sensitive) get more bits, FFN/MoE
 * (robust) get fewer, norms/biases get the most. A "bit ladder" is the sorted
 * set of widths in use; the planner guarantees at least 2 distinct widths
 * (never uniform) and stays within a total-bit budget. Deterministic,
 * no allocation, real work (no stubs). */

typedef enum {
    WUBU_ROLE_EMBED=0,   /* token/positional embedding — most sensitive */
    WUBU_ROLE_ATTN_QKV,  /* attention q/k/v projections */
    WUBU_ROLE_ATTN_OUT,  /* attention output projection */
    WUBU_ROLE_FFN_GATE,  /* FFN gate/up projections */
    WUBU_ROLE_FFN_DOWN,  /* FFN down projection */
    WUBU_ROLE_MOE_EXPERT,/* MoE expert weights — most robust */
    WUBU_ROLE_NORM,      /* layernorm/rmsnorm — keep full precision */
    WUBU_ROLE_BIAS,      /* biases — keep full precision */
    WUBU_ROLE_COUNT
} wubu_role_t;

#define WUBU_BITMIX_MAX_WIDTHS 16

typedef struct {
    int widths[WUBU_ROLE_COUNT];   /* bits per role */
    int budget_bits;               /* total bit budget (sum of widths) */
} wubu_bitmix_t;

/* Plan a per-role bit ladder given a base width and a budget. Sensitive
 * roles get base+1 or more; robust roles get base-1 or fewer; norms/biases
 * get full (e.g. 16). Guarantees >=2 distinct widths (never uniform) and
 * sum(widths) <= budget. Returns 0/-1. */
int wubu_bitmix_plan(wubu_bitmix_t *m, int base_width, int budget_bits);

/* Width assigned to a role. */
int wubu_bitmix_width(const wubu_bitmix_t *m, wubu_role_t role);

/* Number of DISTINCT widths in the ladder (>=2 means non-uniform). */
int wubu_bitmix_distinct(const wubu_bitmix_t *m);

/* Sum of all role widths. */
int wubu_bitmix_total(const wubu_bitmix_t *m);

#endif
