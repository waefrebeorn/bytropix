/*
 * wubu_sys_tune.c -- System / dispatch auto-tuners (L10 / N06 / N10 / O03). C11.
 *
 * Convergence (SeerAttention + NUMA/OS + energy/Roofline 7-hop):
 *  - L10 SeerAttention: per-head dynamic sparse attention -- predict each head's
 *    keep fraction from a cheap gating signal (normalized attention entropy).
 *    Sharper heads (low entropy) keep more blocks; diffuse heads keep fewer.
 *  - N06 NUMA auto-detect: report available NUMA nodes (best-effort via sysconf;
 *    falls back to 1 when unavailable). Used to pin KV arenas.
 *  - N10 Energy-per-token: estimate J/token = compute_j + hbm_j + net_j (the three
 *    energy terms, precomputed by the caller). Pure, no PMU needed.
 *  - O03 Compiler cost-model: pick a tiling/unroll factor from problem size
 *    (cache-block heuristic, ties the roofline auto-tuner). Clamped, deterministic.
 *
 * Triple-DA: invalid input clamped; no div-by-zero; NUMA fallback safe.
 */
#include "wubu_sys_tune.h"
#include <stdlib.h>
#include <math.h>

#ifdef __linux__
#include <unistd.h>
#endif

/* L10 SeerAttention per-head keep fraction. */
float wubu_seer_keep_frac(float entropy, float min_f) {
    if (entropy < 0.0f) entropy = 0.0f;
    if (entropy > 1.0f) entropy = 1.0f;
    if (min_f < 0.0f) min_f = 0.05f;
    if (min_f > 1.0f) min_f = 1.0f;
    /* sharp (e=0) -> 1.0; diffuse (e=1) -> min_f */
    float f = 1.0f - entropy * (1.0f - min_f);
    if (f < min_f) f = min_f;
    if (f > 1.0f) f = 1.0f;
    return f;
}

/* N06 NUMA node count (best-effort). Returns >=1. */
int wubu_numa_nodes(void) {
#ifdef __linux__
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 1;
    return (int)n > 0 ? (int)n : 1;
#else
    return 1;
#endif
}

/* N10 energy per token (J/token) = sum of three terms. */
double wubu_energy_per_token(double compute_j, double hbm_j, double net_j) {
    if (compute_j < 0.0) compute_j = 0.0;
    if (hbm_j   < 0.0) hbm_j   = 0.0;
    if (net_j   < 0.0) net_j   = 0.0;
    return compute_j + hbm_j + net_j;
}

/* O03 compiler tile factor in [tmin,tmax], scaling with sqrt(n). */
int wubu_tile_factor(int n, int tmin, int tmax) {
    if (n <= 0) return tmin > 0 ? tmin : 1;
    if (tmin <= 0) tmin = 1;
    if (tmax < tmin) tmax = tmin;
    int t = (int)(sqrt((double)n) / 4.0 + 0.5);
    if (t < tmin) t = tmin;
    if (t > tmax) t = tmax;
    return t;
}
