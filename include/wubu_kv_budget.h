/*
 * wubu_kv_budget.h -- Layer-wise KV budget / adaptive sink / scheme selector /
 * footprint forecaster (L18 / L19 / N03 / N17). Opaque-free pure functions.
 */
#ifndef WUBU_KV_BUDGET_H
#define WUBU_KV_BUDGET_H

/* L18: fraction of KV cap for layer `layer` (deeper => more). In (0,1]. */
float wubu_layer_kv_budget(int layer, int L, float base,
                           float shallow_frac, float deep_frac);

/* L19: sink count from normalized attention entropy e in [0,1]. */
int wubu_adaptive_sink(float entropy, int min_sink, int max_sink);

/* N03/N17: pick KV bits (b_lo or b_hi) from B* crossover. */
int wubu_kv_scheme_bits(double b_star, int b_lo, int b_hi);

/* N17: projected KV bytes to pre-allocate for (batch, seq) at b_kv. */
double wubu_kv_forecast(int L, int n_kv, int d_h, int b_kv,
                        int batch, int seq);

#endif /* WUBU_KV_BUDGET_H */
