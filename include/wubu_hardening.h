#ifndef WUBU_HARDENING_H
#define WUBU_HARDENING_H
#include "wubu_unit_sandbox.h"

/* wubu_hardening.h -- OMARCHY #283 systemd hardening profile catalog.
 *
 * A catalog of pre-defined systemd hardening profiles (like systemd-analyze
 * security tiers), each composing a set of sandbox directives into a
 * wubu_unit_sandbox_t. Profiles range from minimal (baseline) to strict
 * (maximum confinement). Applying a profile configures a unit's sandbox.
 * Deterministic, real work (no stubs). */

typedef enum {
    WUBU_HARD_BASELINE=0,  /* minimal: ProtectSystem=strict only */
    WUBU_HARD_BALANCED,    /* + ProtectHome, PrivateTmp, NoNewPrivileges */
    WUBU_HARD_STRICT,      /* + RestrictNamespaces, full ProtectHome */
    WUBU_HARD_PARANOID,    /* maximum: everything on, ProtectHome=yes */
    WUBU_HARD_COUNT
} wubu_hard_profile_t;

/* Apply a hardening profile to a sandbox config. Returns 0/-1. */
int wubu_hardening_apply(wubu_unit_sandbox_t *s, wubu_hard_profile_t profile);

/* Profile name. */
const char *wubu_hardening_name(wubu_hard_profile_t profile);

/* A rough security score in [0,100] for a sandbox config (how many
 * hardening directives are on). */
int wubu_hardening_score(const wubu_unit_sandbox_t *s);

#endif
