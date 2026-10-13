#ifndef WUBU_OUTPUT_GUARD_H
#define WUBU_OUTPUT_GUARD_H
#include <stddef.h>

/* wubu_output_guard.h -- AGIEDR #306 LLM-output-as-untrusted validation.
 *
 * Treats every LLM output as UNTRUSTED: before it is acted upon (executed,
 * sent, written), it is validated against a policy. The guard scans the
 * output for dangerous constructs (shell metacharacters, path traversal,
 * embedded commands, credential patterns, control chars) and either PASSES
 * it, or SANITIZES it (strips/escapes the dangerous parts), or REJECTS it
 * (fail-closed). The policy is configurable per strictness. Deterministic,
 * no allocation beyond the caller's buffer, real work (no stubs). */

typedef enum {
    WUBU_OUT_PASS=0,    /* output is safe as-is */
    WUBU_OUT_SANITIZE,  /* output has removable dangerous parts */
    WUBU_OUT_REJECT,    /* output is too dangerous to use (fail-closed) */
} wubu_out_verdict_t;

typedef enum {
    WUBU_OUT_LENIENT=0, /* only reject hard dangers (exec, traversal) */
    WUBU_OUT_STRICT,    /* also sanitize soft dangers (metachars, creds) */
} wubu_out_policy_t;

typedef struct {
    wubu_out_verdict_t verdict;
    int n_dangers;      /* number of dangerous constructs found */
} wubu_out_result_t;

/* Validate an LLM output string. Returns the verdict + danger count. */
wubu_out_result_t wubu_output_guard_check(const char *output, wubu_out_policy_t policy);

/* Sanitize: copy output to dst, stripping/escaping dangerous characters
 * (shell metachars, control chars). Returns bytes written, or -1 on error. */
int wubu_output_guard_sanitize(const char *output, char *dst, size_t dstsz);

#endif
