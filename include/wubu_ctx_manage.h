/*
 * wubu_ctx_manage.h -- Context-window + dispatch auto-managers
 * (L16 / N07 / N14). Pure policy functions.
 */
#ifndef WUBU_CTX_MANAGE_H
#define WUBU_CTX_MANAGE_H

/* L16 elastic window (grow/shrink online by attention entropy). */
int wubu_elastic_window(int W, float entropy, int wmin, int wmax, float rate);

/* N07 tiered-cache advisor -> 0=HOT, 1=WARM, 2=COLD. */
int wubu_tier_advice(float recency, float attn);

/* N14 MoD router calibration -> next gate threshold tau in [0,1]. */
float wubu_mod_tau(float tau, float target, float measured, float lr);

#endif /* WUBU_CTX_MANAGE_H */
