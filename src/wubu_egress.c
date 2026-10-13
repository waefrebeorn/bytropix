/* wubu_egress.c -- AGI-EDR network egress control: block exfiltration.
 * See wubu_egress.h. Default-deny allowlist + per-rule and per-window byte
 * budgets, fail-closed. */
#include "wubu_egress.h"
#include <string.h>

int wubu_egress_init(wubu_egress_t *e, uint64_t window_budget) {
    if (!e) return -1;
    memset(e, 0, sizeof(*e));
    e->window_budget = window_budget;
    return 0;
}

int wubu_egress_allow(wubu_egress_t *e, const char *host, uint16_t port, uint64_t byte_budget) {
    if (!e || !host || e->n_rules >= WUBU_EGRESS_MAX_HOSTS) return -1;
    if (strlen(host) >= WUBU_EGRESS_HOSTLEN) return -1;
    wubu_egress_rule_t *r = &e->rules[e->n_rules++];
    strncpy(r->host, host, WUBU_EGRESS_HOSTLEN - 1);
    r->host[WUBU_EGRESS_HOSTLEN - 1] = '\0';
    r->port = port;
    r->byte_budget = byte_budget;
    r->consumed = 0;
    return 0;
}

/* Host match: exact, or "*." suffix wildcard. Port 0 in a rule = any port. */
static int host_matches(const wubu_egress_rule_t *r, const char *host, uint16_t port) {
    if (r->port != 0 && r->port != port) return 0;
    if (r->host[0] == '*' && r->host[1] == '.') {
        const char *rsuf = r->host + 2;    /* skip "*." -> "example.com" */
        size_t rl = strlen(rsuf);
        size_t hl = strlen(host);
        if (hl < rl) return 0;
        const char *hsuf = host + (hl - rl);
        /* require a domain boundary: exact apex match, or a '.' immediately
         * before the suffix (so "notexample.com" does not match "*.example.com") */
        if (hl > rl && hsuf[-1] != '.') return 0;
        for (size_t i = 0; i < rl; i++) {
            char a = rsuf[i], b = hsuf[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) return 0;
        }
        return 1;
    }
    return strcmp(r->host, host) == 0;
}

static const wubu_egress_rule_t *find_rule(const wubu_egress_t *e, const char *host, uint16_t port) {
    if (!e || !host) return NULL;
    for (int i = 0; i < e->n_rules; i++) {
        if (host_matches(&e->rules[i], host, port)) return &e->rules[i];
    }
    return NULL;
}

int wubu_egress_allowed(const wubu_egress_t *e, const char *host, uint16_t port) {
    const wubu_egress_rule_t *r = find_rule(e, host, port);
    if (!r) return 0;
    /* allowed only if the per-rule budget is not exhausted */
    if (r->consumed >= r->byte_budget) return 0;
    return 1;
}

int wubu_egress_check(wubu_egress_t *e, const char *host, uint16_t port, uint64_t bytes) {
    /* fail-closed: uninitialised controller denies */
    if (!e) return 0;
    const wubu_egress_rule_t *r = find_rule(e, host, port);
    if (!r) { e->denied_count++; return 0; }
    /* find the mutable slot */
    wubu_egress_rule_t *m = NULL;
    for (int i = 0; i < e->n_rules; i++) {
        if (&e->rules[i] == r) { m = &e->rules[i]; break; }
    }
    /* per-rule budget */
    if (m->consumed + bytes > m->byte_budget) { e->denied_count++; return 0; }
    /* global window budget */
    if (e->window_used + bytes > e->window_budget) { e->denied_count++; return 0; }
    /* charge */
    m->consumed += bytes;
    e->window_used += bytes;
    return 1;
}

void wubu_egress_window_reset(wubu_egress_t *e) {
    if (!e) return;
    e->window_used = 0;
}

uint64_t wubu_egress_denied(const wubu_egress_t *e) {
    return e ? e->denied_count : 0;
}
