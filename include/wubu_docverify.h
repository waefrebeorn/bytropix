#ifndef WUBU_DOCVERIFY_H
#define WUBU_DOCVERIFY_H

/* wubu_docverify.h -- STACK #369 docs-vs-code verification.
 *
 * Verifies that documented claims match code: every module name referenced in
 * documentation must have a corresponding implementation (src/wubu_<x>.c) and
 * header (include/wubu_<x>.h). The verifier takes a list of documented module
 * names and a set of "known" module names (those that exist in code), and
 * reports: how many documented modules are backed by code (match), how many
 * are missing (drift), and the match rate. A doc that claims a module that
 * doesn't exist is DRIFT. Deterministic, fixed-size, real work (no stubs). */

#define WUBU_DV_MAX 512

typedef struct {
    char name[64];        /* module name (e.g. "wubu_canary") */
    int  in_docs;         /* referenced in docs */
    int  in_code;         /* exists as src/wubu_<x>.c */
    int  in_header;       /* exists as include/wubu_<x>.h */
} wubu_dv_module_t;

typedef struct {
    wubu_dv_module_t mods[WUBU_DV_MAX];
    int n;
} wubu_docverify_t;

/* Initialise. */
int wubu_dv_init(wubu_docverify_t *v);

/* Record a module with its code/header/doc presence. Returns 0/-1. */
int wubu_dv_record(wubu_docverify_t *v, const char *name,
                   int in_code, int in_header, int in_docs);

/* Count modules that are documented but MISSING from code (drift). */
int wubu_dv_drift(const wubu_docverify_t *v);

/* Count modules fully backed (docs + code + header all present). */
int wubu_dv_backed(const wubu_docverify_t *v);

/* Match rate: backed / documented (in [0,1]). 1.0 = docs fully match code. */
double wubu_dv_match_rate(const wubu_docverify_t *v);

/* Does the docs match the code? (drift == 0). 1/0. */
int wubu_dv_consistent(const wubu_docverify_t *v);

#endif
