#ifndef WUBU_NNP_H
#define WUBU_NNP_H
#include "wubu_unit_sandbox.h"

/* wubu_nnp.h -- OMARCHY #284 NoNewPrivileges + PrivateTmp default.
 *
 * A hardening enforcer that forces NoNewPrivileges=yes and PrivateTmp=yes ON
 * by default and REJECTS any attempt to disable them. These two are the
 * baseline non-negotiable sandbox directives: NoNewPrivileges blocks
 * setuid/privilege escalation, PrivateTmp isolates /tmp. The enforcer audits
 * a proposed sandbox config and either passes it (both on) or fails-closed
 * (either off -> reject). Deterministic, real work (no stubs). */

/* Enforce the defaults: force NoNewPrivileges + PrivateTmp on. Returns 0. */
int wubu_nnp_enforce(wubu_unit_sandbox_t *s);

/* Audit a config: returns 1 if both mandatory directives are on (compliant),
 * 0 if either is off (non-compliant, fail-closed). */
int wubu_nnp_audit(const wubu_unit_sandbox_t *s);

/* Can a unit request to DISABLE NoNewPrivileges? Always 0 (non-negotiable). */
int wubu_nnp_may_disable(void);

/* Can a unit request to disable PrivateTmp? Always 0 (non-negotiable). */
int wubu_nnp_may_disable_tmp(void);

#endif
