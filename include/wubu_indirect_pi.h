#ifndef WUBU_INDIRECT_PI_H
#define WUBU_INDIRECT_PI_H
#include <stddef.h>

/* wubu_indirect_pi.h -- AGIEDR #304 indirect prompt-injection defense.
 *
 * Indirect prompt injection: malicious instructions hidden in UNTRUSTED
 * content the agent reads (tool outputs, retrieved documents, web pages,
 * emails) rather than in the direct user prompt. The defense scans retrieved
 * content for injection payloads (imperative overrides, hidden instructions,
 * role reassignment) BEFORE it enters the agent's context, and either
 * sanitizes or quarantines it. Distinct from direct injection (wubu_inject):
 * this guards the READ path. Fail-closed, deterministic, real work. */

typedef enum {
    WUBU_IPI_CLEAN=0,   /* content is safe to read */
    WUBU_IPI_SUSPECT,   /* possible injection -- sanitize before use */
    WUBU_IPI_MALICIOUS, /* clear injection -- quarantine */
} wubu_ipi_verdict_t;

/* Scan untrusted content for indirect-injection payloads. Returns the
 * verdict. NULL -> MALICIOUS (fail-closed). */
wubu_ipi_verdict_t wubu_indirect_pi_scan(const char *content);

/* Should this content be blocked from entering the agent's context? 1/0. */
int wubu_indirect_pi_block(const char *content);

/* A risk score in [0,1] for the content. */
double wubu_indirect_pi_score(const char *content);

#endif
