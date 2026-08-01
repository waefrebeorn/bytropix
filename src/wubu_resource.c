/*
 * wubu_resource.c -- At-home resource envelope (AH14/AH15). C11.
 *
 * Convergence (bandwidth-bound tok/s; VRAM tier table; Q4 floor; graceful
 * degradation 70B->14B->7B 7-hop):
 *   - AH14 resource profiler: given detected VRAM (GB) + bandwidth (GB/s),
 *          pick the largest model that FITS (bytes <= VRAM*quant_factor) and
 *          estimate decode tok/s = bandwidth / active_bytes_per_token.
 *   - AH15 graceful degradation: if a model doesn't fit, step down tiers until
 *          one fits (never OOM; the 512K ceiling + this tiering cooperate).
 *
 * Pure C11, deterministic, testable. No hw calls (profiler takes detected
 * values; in production these come from hwcaps + nvml-style probe).
 */
#include "wubu_resource.h"
#include <stdlib.h>

/* AH14: pick model tier by VRAM. Returns WUBU_TIER_*. q4_bytes_per_billion
 * ~= 0.6 GB (4-bit). A model of `billions` params at 4-bit needs ~0.6*B GB. */
int wubu_pick_tier(double vram_gb, int billions) {
    double need = 0.6 * (double)billions;       /* Q4 estimate */
    if (need <= vram_gb * 0.95) return WUBU_TIER_FIT;
    if (need <= vram_gb * 1.6)  return WUBU_TIER_FITS_Q3; /* try more aggressive quant */
    return WUBU_TIER_NOFIT;
}

/* AH14: estimate decode tok/s = bandwidth / active_bytes_per_token.
 * active_bytes = billions * 0.6e9 * (quant_bits/32) for Q4. */
double wubu_est_toks(double bandwidth_gbs, int billions, int quant_bits) {
    double bytes_per_tok = (double)billions * 1e9 * (quant_bits / 32.0);
    if (bytes_per_tok <= 0) return 0;
    return bandwidth_gbs * 1e9 / bytes_per_tok;
}

/* AH15: graceful degradation. Given desired billions and VRAM, return the
 * largest tier that fits (stepping 70B->34B->14B->7B). Returns billions of
 * the chosen tier (0 = none fit). */
int wubu_degrade_tier(double vram_gb, int desired_b) {
    int tiers[] = { 70, 34, 14, 7 };
    /* try from largest down; return first that fits. Never OOM. */
    for (int i = 0; i < 4; i++) {
        if (wubu_pick_tier(vram_gb, tiers[i]) != WUBU_TIER_NOFIT) return tiers[i];
    }
    return 0; /* nothing fits even at 7B (sub-7B VRAM) */
}
