/*
 * wubu_sys_tune.h -- System / dispatch auto-tuners (L10/N06/N10/O03).
 */
#ifndef WUBU_SYS_TUNE_H
#define WUBU_SYS_TUNE_H

/* L10 SeerAttention per-head keep fraction in [min_f, 1] from entropy. */
float wubu_seer_keep_frac(float entropy, float min_f);

/* N06 NUMA node count (best-effort, >=1). */
int wubu_numa_nodes(void);

/* N10 energy per token (J/token) = sum of three non-negative terms. */
double wubu_energy_per_token(double compute_j, double hbm_j, double net_j);

/* O03 compiler tile factor in [tmin, tmax] from problem size n. */
int wubu_tile_factor(int n, int tmin, int tmax);

#endif /* WUBU_SYS_TUNE_H */
