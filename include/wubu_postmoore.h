#ifndef WUBU_POSTMOORE_H
#define WUBU_POSTMOORE_H

/* wubu_postmoore.h -- PATHS #328 post-Moore algorithmic-scaling research.
 *
 * Post-Moore scaling: with transistor scaling slowing, capability gains come
 * from ALGORITHMS that extract more from the same silicon -- not more FLOPs.
 * This module models an efficiency stack: each algorithmic technique
 * contributes a compute multiplier for the SAME capability, and the combined
 * speedup is the product. Techniques modeled (all implemented in this repo):
 *   - custom ISA (tighter codegen than a portable ABI path)
 *   - speculative decoding (draft + verify: ~acceptance-rate speedup)
 *   - hyperbolic MoE routing (sparse expert activation)
 *   - memory consolidation (sleep-replay: cheaper than retraining)
 * The model reports the total efficiency multiplier and the equivalent
 * "algorithmic years" of transistor scaling gained. Deterministic, real work
 * (no stubs). */

typedef struct {
    double custom_isa;        /* speedup from the custom-ISA codegen path */
    double spec_decode;       /* speedup from speculative decoding */
    double moe_routing;       /* speedup from sparse MoE routing */
    double mem_consolidation; /* speedup from replay vs retraining */
} wubu_postmoore_t;

/* Initialise with the four technique multipliers (each >= 1.0). Returns 0/-1. */
int wubu_pm_init(wubu_postmoore_t *p, double custom_isa, double spec_decode,
                 double moe_routing, double mem_consolidation);

/* Combined efficiency multiplier (product of all techniques). */
double wubu_pm_total_speedup(const wubu_postmoore_t *p);

/* Compute saved for a fixed capability (fraction reduced, in [0,1)). */
double wubu_pm_compute_saved(const wubu_postmoore_t *p);

/* Effective capability at fixed compute (multiplier), i.e. how much more the
 * same silicon can do. */
double wubu_pm_effective_capability(const wubu_postmoore_t *p);

/* "Algorithmic years" gained: Moore's law ~doubles transistors every 2 years,
 * so an N-x efficiency gain is worth log2(N) doublings = log2(N)*2 years of
 * transistor scaling. */
double wubu_pm_algorithmic_years(const wubu_postmoore_t *p);

#endif
