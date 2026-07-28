#ifndef WUBU_KEREQ_H
#define WUBU_KEREQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An affine+clamp kernel: y = clamp(scale*x + bias, lo, hi). */
typedef struct {
    float scale;
    float bias;
    float lo;   /* clamp lower bound */
    float hi;   /* clamp upper bound */
} wubu_affine_clamp_t;

/* Prove equivalence of ref vs cand over input range [xlo, xhi] by abstract
 * interpretation (interval arithmetic, sound).
 * Returns:
 *   1 = proven EQUAL (UNSAT)
 *   0 = proven DIVERGENT (SAT); *cx = counterexample output value
 *   2 = UNKNOWN (intervals overlap, inconclusive) */
int wubu_kereq_prove_eq(const wubu_affine_clamp_t *ref,
                        const wubu_affine_clamp_t *cand,
                        float xlo, float xhi, float *cx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KEREQ_H */
