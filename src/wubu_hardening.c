/* wubu_hardening.c -- OMARCHY #283 systemd hardening profile catalog.
 * See wubu_hardening.h. */
#include "wubu_hardening.h"
#include <string.h>

const char *wubu_hardening_name(wubu_hard_profile_t profile) {
    switch (profile) {
        case WUBU_HARD_BASELINE: return "baseline";
        case WUBU_HARD_BALANCED: return "balanced";
        case WUBU_HARD_STRICT:   return "strict";
        case WUBU_HARD_PARANOID: return "paranoid";
        default:                 return "unknown";
    }
}

int wubu_hardening_apply(wubu_unit_sandbox_t *s, wubu_hard_profile_t profile) {
    if (!s || profile < 0 || profile >= WUBU_HARD_COUNT) return -1;
    memset(s, 0, sizeof *s);
    /* every profile: ProtectSystem=strict (whole hierarchy read-only) */
    s->protect_system = WUBU_PS_STRICT;
    switch (profile) {
        case WUBU_HARD_BASELINE:
            s->protect_home = WUBU_PH_NO;
            s->private_tmp = 0;
            s->no_new_privileges = 0;
            s->restrict_namespaces = 0;
            break;
        case WUBU_HARD_BALANCED:
            s->protect_home = WUBU_PH_READONLY;
            s->private_tmp = 1;
            s->no_new_privileges = 1;
            s->restrict_namespaces = 0;
            break;
        case WUBU_HARD_STRICT:
            s->protect_home = WUBU_PH_READONLY;
            s->private_tmp = 1;
            s->no_new_privileges = 1;
            s->restrict_namespaces = 1;
            break;
        case WUBU_HARD_PARANOID:
            s->protect_home = WUBU_PH_YES;   /* home fully inaccessible */
            s->private_tmp = 1;
            s->no_new_privileges = 1;
            s->restrict_namespaces = 1;
            break;
        default: break;
    }
    return 0;
}

int wubu_hardening_score(const wubu_unit_sandbox_t *s) {
    if (!s) return 0;
    int score = 0;
    if (s->protect_system == WUBU_PS_STRICT) score += 25;
    else if (s->protect_system == WUBU_PS_FULL) score += 30;
    if (s->protect_home == WUBU_PH_READONLY) score += 20;
    else if (s->protect_home == WUBU_PH_YES) score += 30;
    if (s->private_tmp) score += 15;
    if (s->no_new_privileges) score += 20;
    if (s->restrict_namespaces) score += 15;
    if (score > 100) score = 100;
    return score;
}
