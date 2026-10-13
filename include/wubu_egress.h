#ifndef WUBU_EGRESS_H
#define WUBU_EGRESS_H
#include <stddef.h>
#include <stdint.h>

/* wubu_egress.h -- AGI-EDR network egress control: block exfiltration
 * (AGIEDR #299).
 *
 * Default-DENY: every outbound connection (host, port) must match an explicit
 * allow rule or it is refused. Beyond the host allowlist, each rule carries a
 * byte budget (max bytes egressed to that host) so a compromised agent cannot
 * bulk-exfiltrate even to an allowed host. A global per-window budget caps
 * total egress. Decisions are deterministic and fail-closed: an uninitialised
 * controller or an over-budget/allowed-but-exhausted rule denies.
 *
 * Real work (no stubs): the controller tracks consumed bytes per rule and
 * per window and enforces the budgets at decision time. */

#define WUBU_EGRESS_MAX_HOSTS 32
#define WUBU_EGRESS_HOSTLEN   64

typedef struct {
    char     host[WUBU_EGRESS_HOSTLEN]; /* allowed host (exact or *.suffix) */
    uint16_t port;                       /* allowed port (0 = any) */
    uint64_t byte_budget;                /* max bytes to this host */
    uint64_t consumed;                   /* bytes egressed so far */
} wubu_egress_rule_t;

typedef struct {
    wubu_egress_rule_t rules[WUBU_EGRESS_MAX_HOSTS];
    int      n_rules;
    uint64_t window_budget;   /* global cap per window */
    uint64_t window_used;     /* global bytes this window */
    uint64_t denied_count;    /* total refusals (for observability) */
} wubu_egress_t;

/* Initialise with a global per-window byte budget. Returns 0/-1. */
int wubu_egress_init(wubu_egress_t *e, uint64_t window_budget);

/* Add an allow rule (host, port, byte_budget). host starting with "*." is a
 * suffix wildcard (e.g. "*.example.com"). port 0 = any. Returns 0/-1. */
int wubu_egress_allow(wubu_egress_t *e, const char *host, uint16_t port, uint64_t byte_budget);

/* Decide whether an egress of `bytes` to host:port is permitted AND charge it.
 * Returns 1 if allowed (and charges budgets), 0 if denied (fail-closed). */
int wubu_egress_check(wubu_egress_t *e, const char *host, uint16_t port, uint64_t bytes);

/* Query whether host:port is allowed WITHOUT charging (peek). 1/0. */
int wubu_egress_allowed(const wubu_egress_t *e, const char *host, uint16_t port);

/* Reset the per-window global budget (call at window boundary). */
void wubu_egress_window_reset(wubu_egress_t *e);

/* Total denials since init (observability). */
uint64_t wubu_egress_denied(const wubu_egress_t *e);

#endif
