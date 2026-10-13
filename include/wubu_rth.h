#ifndef WUBU_RT_HARNESS_H
#define WUBU_RT_HARNESS_H
#include <stdint.h>

/* wubu_rt_harness.h -- AGIEDR #310 adversarial red-team agent harness.
 *
 * An automated red-team harness: an adversarial agent generates ATTACK VARIANTS
 * by mutating a seed attack (insert/append/obfuscate tokens) and probes a
 * defense oracle with each. It records which variants the defense missed
 * (attack succeeded) -- these are the bypasses to fix. The harness iterates,
 * mutating successful bypasses to explore the attack space. Deterministic,
 * real work (no stubs). */

#define WUBU_RTH_MAX_PROBES 64
#define WUBU_RTH_ATKLEN 96

typedef struct {
    char attack[WUBU_RTH_ATKLEN];
    int  blocked;      /* 1 = defense caught it, 0 = BYPASS (defense missed) */
} wubu_rth_probe_t;

/* The defense oracle under test: returns 1 if the attack is blocked. */
typedef int (*wubu_rth_defense_t)(const char *attack, void *ctx);

typedef struct {
    wubu_rth_probe_t probes[WUBU_RTH_MAX_PROBES];
    int n;
    int n_bypasses;    /* attacks the defense missed */
    uint32_t rng;
} wubu_rth_t;

/* Initialise with an RNG seed. */
int wubu_rth_init(wubu_rth_t *h, uint32_t seed);

/* Mutate a seed attack (insert/append/obfuscate a token) into out. Returns
 * the length written, or -1. */
int wubu_rth_mutate(wubu_rth_t *h, const char *seed, char *out, int cap);

/* Run one probe: mutate the seed, test the defense, record the result.
 * Returns 1 if the attack was BLOCKED, 0 if it BYPASSED, -1 on error. */
int wubu_rth_probe(wubu_rth_t *h, const char *seed, wubu_rth_defense_t def, void *ctx);

/* Number of bypasses found (attacks the defense missed). */
int wubu_rth_bypasses(const wubu_rth_t *h);

/* Bypass rate: bypasses / total probes (in [0,1]). */
double wubu_rth_bypass_rate(const wubu_rth_t *h);

#endif
