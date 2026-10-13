#ifndef WUBU_DGMSAFETY_H
#define WUBU_DGMSAFETY_H

/* wubu_dgmsafety.h -- PATHS #326 DGM safety-precautions verified.
 *
 * The Darwin Godel Machine rewrites its own code; before ANY self-modification
 * is applied, it must be checked against declared SAFETY PRECAUTIONS --
 * invariants that must continue to hold (e.g. "no new syscalls", "memory
 * bounded", "core tests still pass", "capability set unchanged"). A
 * modification that violates any precaution is REJECTED (fail-closed). This is
 * the verified-safety gate on self-modification. Deterministic, real work
 * (no stubs). */

#define WUBU_DGM_MAX_PRECAUTIONS 16
#define WUBU_DGM_TEXTLEN 64

typedef struct {
    char name[WUBU_DGM_TEXTLEN];
    int holds;    /* 1 = this invariant currently holds */
} wubu_dgm_precaution_t;

typedef struct {
    wubu_dgm_precaution_t items[WUBU_DGM_MAX_PRECAUTIONS];
    int n;
    int n_checked;
    int n_violated;
} wubu_dgmsafety_t;

/* Initialise. */
int wubu_dgm_init(wubu_dgmsafety_t *s);

/* Declare a safety precaution (an invariant that must hold). Returns 0/-1. */
int wubu_dgm_declare(wubu_dgmsafety_t *s, const char *name);

/* Verify a proposed self-modification: for each precaution, 'holds' says
 * whether it still holds under the modification. Returns 1 if SAFE (all hold),
 * 0 if REJECTED (any violated, fail-closed). */
int wubu_dgm_verify(wubu_dgmsafety_t *s, const int *holds, int n);

/* Was the last verification safe? 1/0. */
int wubu_dgm_last_safe(const wubu_dgmsafety_t *s);

/* Total violations detected across all verifications. */
int wubu_dgm_violations(const wubu_dgmsafety_t *s);

#endif
