/*
 * wubu_verify.c -- C11 type-check + invariant assertion gate (AX09). C11.
 *
 * Convergence (formal verification 7-hop: SAFE, rocc-of-rust, Aeneas, Lean):
 *   - AX09: a lightweight formal gate — assertion-based invariant checking
 *     on generated code before it enters the decode path. This is the
 *     pragmatic home-scale approximation of proof-checking.
 */
#include "wubu_verify.h"
#include <stdio.h>
#include <string.h>

int wubu_verify_init(wubu_verify_t *v) {
    if (!v) return -1;
    v->n_asserts = 0;
    v->n_passed = 0;
    v->n_failed = 0;
    return 0;
}

int wubu_verify_assert_int(wubu_verify_t *v, int cond, const char *expr_str) {
    if (!v || !expr_str) return -1;
    if (v->n_asserts >= WUBU_VERIFY_MAX_ASSERTS) return -1;
    v->n_asserts++;
    if (cond) { v->n_passed++; }
    else { v->n_failed++; }
    return cond ? 1 : 0;
}

int wubu_verify_assert_ptr(wubu_verify_t *v, const void *ptr, const char *expr_str) {
    if (!v) return -1;
    return wubu_verify_assert_int(v, ptr != NULL, expr_str);
}

int wubu_verify_assert_range(wubu_verify_t *v, long val, long lo, long hi, const char *expr_str) {
    if (!v) return -1;
    return wubu_verify_assert_int(v, val >= lo && val < hi, expr_str);
}

int wubu_verify_all_passed(const wubu_verify_t *v) {
    if (!v) return 0;
    return (v->n_failed == 0 && v->n_asserts > 0) ? 1 : 0;
}

int wubu_verify_count(const wubu_verify_t *v, int *out_passed, int *out_failed) {
    if (!v || !out_passed || !out_failed) return -1;
    *out_passed = v->n_passed;
    *out_failed = v->n_failed;
    return v->n_asserts;
}