/*
 * wubu_smt_check.h -- SMT-style equivalence checking of GEMV rewrites (doc F02).
 *
 * Bounded exhaustive verification that quantized GEMV matches the
 * reference F32 dot product within a tolerance.
 *
 * Basis: Alive2 (Lopes et al., PLDI 2021); arXiv:2511.12638.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_SMT_CHECK_H
#define WUBU_SMT_CHECK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_SMT_OK          = 0,
    WUBU_SMT_FAIL        = 1,
    WUBU_SMT_UNSUPPORTED = 2
} wubu_smt_status_t;

typedef struct {
    wubu_smt_status_t status;
    int n_checks;       /* total checks performed */
    int n_failures;     /* number of failures */
    int8_t first_fail_w;  /* first failing weight value */
    int8_t first_fail_x;  /* first failing activation value */
    float max_error;    /* maximum error observed */
} wubu_smt_result_t;

/* Run bounded exhaustive check on quantized GEMV for given K.
 * Tests boundary/extreme value combinations + mixed patterns.
 * K must be <= 16 for tractable exhaustive verification. */
wubu_smt_result_t wubu_smt_check_gemv(int K, float tolerance);

/* Verify a specific (W_q, x_q, scales) tuple. */
wubu_smt_result_t wubu_smt_verify_specific(const int8_t *w_q, const int8_t *x_q,
                                             float w_scale, float x_scale,
                                             int K, float tolerance);

/* Get status string. */
const char *wubu_smt_status_str(wubu_smt_status_t s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SMT_CHECK_H */
