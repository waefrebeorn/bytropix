#ifndef WUBU_KEREQ_H
#define WUBU_KEREQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prove two clamp(scale*x+bias) kernels equal over [x_lo, x_hi].
 * Returns 1 = proven equal (UNSAT), 0 = divergence found (SAT, *cx = counterexample). */
int wubu_kereq_prove_eq(float x_lo, float x_hi, float scale, float bias,
                        float clamp_lo, float clamp_hi, int buggy, float *cx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KEREQ_H */
