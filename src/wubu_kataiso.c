/* wubu_kataiso.c -- OMARCHY #290 gVisor/Kata-style kernel isolation.
 * See wubu_kataiso.h. Real, deterministic sentry/boundary engine. */

#include "wubu_kataiso.h"
#include <string.h>

const char *wubu_iso_action_name(wubu_iso_action_t a)
{
    switch (a) {
    case WUBU_ISO_PASSTHROUGH: return "passthrough";
    case WUBU_ISO_EMULATE:     return "emulate";
    case WUBU_ISO_DENY:        return "deny";
    default:                   return "unknown";
    }
}

int wubu_iso_init(wubu_kataiso_t *b)
{
    if (!b) return -1;
    memset(b, 0, sizeof(*b));
    return 0;
}

int wubu_iso_classify(wubu_kataiso_t *b, int syscall_nr, wubu_iso_action_t action)
{
    if (!b || b->sealed) return -1;
    if (action != WUBU_ISO_PASSTHROUGH && action != WUBU_ISO_EMULATE &&
        action != WUBU_ISO_DENY) return -1;
    /* override any existing rule for this nr */
    for (int i = 0; i < b->n_rules; i++) {
        if (b->rules[i].syscall_nr == syscall_nr) {
            b->rules[i].action = action;
            return 0;
        }
    }
    if (b->n_rules >= WUBU_ISO_MAX_RULES) return -1;
    b->rules[b->n_rules].syscall_nr = syscall_nr;
    b->rules[b->n_rules].action = action;
    b->n_rules++;
    return 0;
}

int wubu_iso_seal(wubu_kataiso_t *b)
{
    if (!b) return -1;
    if (b->sealed) return -1;
    b->sealed = 1;
    return 0;
}

wubu_iso_action_t wubu_iso_lookup(const wubu_kataiso_t *b, int syscall_nr)
{
    if (!b) return WUBU_ISO_DENY;
    for (int i = 0; i < b->n_rules; i++)
        if (b->rules[i].syscall_nr == syscall_nr)
            return b->rules[i].action;
    return WUBU_ISO_DENY;   /* unclassified -> fail-closed */
}

wubu_iso_action_t wubu_iso_cross(wubu_kataiso_t *b, int syscall_nr)
{
    if (!b) return WUBU_ISO_DENY;
    wubu_iso_action_t a = wubu_iso_lookup(b, syscall_nr);
    b->n_total++;
    switch (a) {
    case WUBU_ISO_PASSTHROUGH: b->n_passthrough++; break;
    case WUBU_ISO_EMULATE:     b->n_emulate++;     break;
    default:                   b->n_deny++;        break;
    }
    return a;
}

int wubu_iso_deny_per_mille(const wubu_kataiso_t *b)
{
    if (!b || b->n_total == 0) return 0;
    return (int)((b->n_deny * 1000) / b->n_total);
}
