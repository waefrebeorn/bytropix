#ifndef WUBU_KATAISO_H
#define WUBU_KATAISO_H

/* wubu_kataiso.h -- OMARCHY #290 gVisor/Kata-style kernel isolation.
 *
 * A user/kernel isolation boundary ("sentry") model, as in gVisor/Kata: a
 * guest process is separated from the host kernel by a boundary that mediates
 * EVERY syscall. The sentry classifies each syscall as one of:
 *   PASSTHROUGH -- forwarded to the host kernel (allowlisted, safe)
 *   EMULATE    -- handled by the sentry itself (never reaches the host kernel)
 *   DENY       -- blocked at the boundary (fail-closed)
 * An unclassified syscall is DENIED. The boundary tracks crossings so a guest
 * cannot silently bypass it. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_ISO_MAX_RULES 256

typedef enum {
    WUBU_ISO_PASSTHROUGH = 0,
    WUBU_ISO_EMULATE = 1,
    WUBU_ISO_DENY = 2
} wubu_iso_action_t;

typedef struct {
    int syscall_nr;
    wubu_iso_action_t action;
} wubu_iso_rule_t;

typedef struct {
    wubu_iso_rule_t rules[WUBU_ISO_MAX_RULES];
    int n_rules;
    /* boundary crossing counters */
    uint64_t n_passthrough;
    uint64_t n_emulate;
    uint64_t n_deny;
    uint64_t n_total;
    int sealed;   /* once sealed, no new rules may be added */
} wubu_kataiso_t;

/* Initialise an empty (default-deny) isolation boundary. Returns 0. */
int wubu_iso_init(wubu_kataiso_t *b);

/* Classify a syscall. Returns 0. Later rules for the same nr override. */
int wubu_iso_classify(wubu_kataiso_t *b, int syscall_nr, wubu_iso_action_t action);

/* Seal the boundary: no further rules accepted (returns -1 if already
 * sealed). After sealing, wubu_iso_cross uses the frozen rule set. */
int wubu_iso_seal(wubu_kataiso_t *b);

/* Cross the boundary with `syscall_nr`. Returns the action taken (the rule's
 * action, or DENY if unclassified/NULL boundary) and updates the counters. */
wubu_iso_action_t wubu_iso_cross(wubu_kataiso_t *b, int syscall_nr);

/* Query the configured action for a syscall without crossing. Returns DENY
 * if unclassified. */
wubu_iso_action_t wubu_iso_lookup(const wubu_kataiso_t *b, int syscall_nr);

/* Fraction of crossings that were denied, in per-mille (0..1000). Returns 0
 * if no crossings yet. */
int wubu_iso_deny_per_mille(const wubu_kataiso_t *b);

/* Human-readable action name. */
const char *wubu_iso_action_name(wubu_iso_action_t a);

#endif
