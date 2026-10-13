/* wubu_edrcard.c -- AGIEDR #311 AGI-EDR model card + disclosure.
 * See wubu_edrcard.h. */
#include "wubu_edrcard.h"
#include <string.h>
#include <stdio.h>

int wubu_edr_init(wubu_edrcard_t *c, const char *threat_model) {
    if (!c || !threat_model) return -1;
    memset(c, 0, sizeof *c);
    if (strlen(threat_model) >= WUBU_EDR_TEXTLEN) return -1;
    strncpy(c->threat_model, threat_model, WUBU_EDR_TEXTLEN - 1);
    c->n_items = 0;
    return 0;
}

int wubu_edr_add(wubu_edrcard_t *c, wubu_edr_kind_t kind, const char *text) {
    if (!c || !text || c->n_items >= WUBU_EDR_MAX_ITEMS) return -1;
    if (kind < WUBU_EDR_DETECTS || kind > WUBU_EDR_RESPONSE) return -1;
    if (strlen(text) >= WUBU_EDR_TEXTLEN) return -1;
    strncpy(c->items[c->n_items], text, WUBU_EDR_TEXTLEN - 1);
    c->kinds[c->n_items] = kind;
    c->n_items++;
    return 0;
}

int wubu_edr_count(const wubu_edrcard_t *c, wubu_edr_kind_t kind) {
    if (!c) return 0;
    int n = 0;
    for (int i = 0; i < c->n_items; i++) if (c->kinds[i] == kind) n++;
    return n;
}

int wubu_edr_render(const wubu_edrcard_t *c, char *out, int cap) {
    if (!c || !out || cap <= 0) return -1;
    int len = 0;
    len += snprintf(out + len, cap - len, "# AGI-EDR Security Disclosure\n\n");
    len += snprintf(out + len, cap - len, "Threat model: %s\n\n", c->threat_model);
    const char *titles[3] = { "Detects", "Known blind spots", "Response actions" };
    for (int k = 0; k < 3; k++) {
        len += snprintf(out + len, cap - len, "## %s\n", titles[k]);
        int any = 0;
        for (int i = 0; i < c->n_items; i++) {
            if (c->kinds[i] == k) {
                len += snprintf(out + len, cap - len, "- %s\n", c->items[i]);
                any = 1;
            }
        }
        if (!any) len += snprintf(out + len, cap - len, "- (none disclosed)\n");
        len += snprintf(out + len, cap - len, "\n");
    }
    return len;
}

int wubu_edr_disclosable(const wubu_edrcard_t *c) {
    if (!c) return 0;
    /* must disclose at least one detection AND one blind spot (honesty gate) */
    return (wubu_edr_count(c, WUBU_EDR_DETECTS) >= 1 &&
            wubu_edr_count(c, WUBU_EDR_BLINDSPOT) >= 1) ? 1 : 0;
}
