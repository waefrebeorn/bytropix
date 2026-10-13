/* wubu_nnp.c -- OMARCHY #284 NoNewPrivileges + PrivateTmp default.
 * See wubu_nnp.h. */
#include "wubu_nnp.h"

int wubu_nnp_enforce(wubu_unit_sandbox_t *s) {
    if (!s) return -1;
    s->no_new_privileges = 1;   /* force on */
    s->private_tmp = 1;         /* force on */
    return 0;
}

int wubu_nnp_audit(const wubu_unit_sandbox_t *s) {
    if (!s) return 0;   /* fail-closed */
    return (s->no_new_privileges && s->private_tmp) ? 1 : 0;
}

int wubu_nnp_may_disable(void) {
    return 0;   /* non-negotiable */
}

int wubu_nnp_may_disable_tmp(void) {
    return 0;   /* non-negotiable */
}
